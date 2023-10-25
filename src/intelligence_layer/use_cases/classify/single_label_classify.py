import math
import re
from typing import (
    Iterable,
    Mapping,
    Optional,
    Sequence,
)

from aleph_alpha_client import (
    Client,
    PromptTemplate,
    Prompt,
)
from pydantic import BaseModel

from intelligence_layer.core.completion import (
    RawCompletion,
)
from intelligence_layer.core.echo import EchoInput, EchoTask, TokenWithProb
from intelligence_layer.task import (
    Chunk,
    Evaluator,
    Probability,
    Task,
    DebugLogger,
    Token,
)


def to_aa_tokens_prompt(tokens: Sequence[Token]) -> Prompt:
    return Prompt.from_tokens([token.token_id for token in tokens])


class ClassifyInput(BaseModel):
    """Input for a classification task.

    Attributes:
        chunk: text to be classified.
        labels: Possible labels the model will choose a label from
    """

    chunk: Chunk
    labels: frozenset[str]


class ClassifyOutput(BaseModel):
    """Output for a single label classification task.

    Attributes:
        scores: Mapping of the provided label (key) to corresponding score (value).
            The score represents how sure the model is that this is the correct label.
            This will be a value between 0 and 1.
            The sum of all probabilities will be 1.
    """

    scores: Mapping[str, Probability]


class SingleLabelClassify(Task[ClassifyInput, ClassifyOutput]):
    """Task that classifies a given input text with one of the given classes.

    The input contains a complete set of all possible labels. The output will return a score for
    each possible label. All scores will add up to 1 and are relative to each other. The highest
    score is given to the most likely class.

    This methodology works best for classes that are easily understood, and don't require an
    explanation or examples.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Attributes:
        PROMPT_TEMPLATE_STR: The prompt template used for answering the question.
            'text' and 'labels' will be inserted here.
        MODEL: A valid Aleph Alpha model name.

    Example:
        >>> client = Client(token="AA_TOKEN")
        >>> task = SingleLabelClassify(client)
        >>> input = SingleLabelClassifyInput(
                text="This is a happy text.",
                labels={"positive", "negative"}
            )
        >>> logger = InMemoryLogger(name="Classify")
        >>> output = task.run(input, logger)
        >>> print(output.scores["positive"])
        0.9
    """

    PROMPT_TEMPLATE: str = """### Instruction:
Identify a class that describes the text adequately.
Reply with only the class label.

### Input:
{{text}}

### Response:"""
    MODEL: str = "luminous-base-control"
    _client: Client

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._client = client
        self._completion_task = RawCompletion(client)
        self._echo_task = EchoTask(client)

    def run(self, input: ClassifyInput, logger: DebugLogger) -> ClassifyOutput:
        log_probs_per_label = self._log_probs_per_label(
            text_to_classify=input.chunk,
            labels=input.labels,
            model=self.MODEL,
            logger=logger,
        )
        logger.log("Log probs per label", log_probs_per_label)
        normalized_probs_per_label = self._normalize(log_probs_per_label, logger)
        scores = self._compute_scores(normalized_probs_per_label)
        return ClassifyOutput(
            scores=scores,
        )

    def _log_probs_per_label(
        self,
        text_to_classify: str,
        labels: frozenset[str],
        model: str,
        logger: DebugLogger,
    ) -> Mapping[str, Sequence[TokenWithProb]]:
        prompt = PromptTemplate(template_str=self.PROMPT_TEMPLATE).to_prompt(
            text=text_to_classify
        )
        return {
            label: self._echo_task.run(
                EchoInput(
                    prompt=prompt,
                    expected_completion=self._prefix_with_whitespace(label),
                    model=model,
                ),
                logger,
            ).tokens_with_log_probs
            for label in labels
        }

    def _prefix_with_whitespace(self, label: str) -> str:
        label = label if re.match(r"^\s+", label) else f" {label}"
        return label + "<|endoftext|>"

    def _compute_scores(
        self,
        normalized_probs_per_score: Mapping[str, Sequence[TokenWithProb]],
    ) -> Mapping[str, Probability]:
        return {
            label: Probability(
                math.prod(token_with_prob.prob for token_with_prob in tokens_with_probs)
            )
            for label, tokens_with_probs in normalized_probs_per_score.items()
        }

    def _normalize(
        self,
        log_probs_per_label: Mapping[str, Sequence[TokenWithProb]],
        logger: DebugLogger,
    ) -> Mapping[str, Sequence[TokenWithProb]]:
        node = TreeNode()
        for log_probs in log_probs_per_label.values():
            node.insert_path(log_probs)

        node.normalize_probs()
        normalized_probs = {
            label: list(
                node.path(
                    token_with_prob.token
                    for token_with_prob in log_probs_per_label[label]
                )
            )
            for label in log_probs_per_label
        }
        logger.log("Normalized Probs", normalized_probs)
        return normalized_probs


class TreeNode:
    def __init__(
        self, token: Optional[Token] = None, prob: Optional[Probability] = None
    ):
        self.token = token
        self.prob = prob
        self.normalized_prob: Optional[Probability] = None
        self.children: list[TreeNode] = []

    def find_child(self, token: Token) -> Optional["TreeNode"]:
        return next((child for child in self.children if child.token == token), None)

    def insert_without_calculation(self, path: Sequence[TokenWithProb]) -> None:
        """Inserts a path into the tree without changing the original probability

        Temporarily here until we change this data structure to be more versatile"""
        if not path:
            return
        token_with_prob = path[0]
        child = self.find_child(token_with_prob.token)
        if child is None:
            child = TreeNode(token_with_prob.token, Probability(token_with_prob.prob))
            self.children.append(child)

        child.insert_without_calculation(path[1:])

    def insert_path(self, path: Sequence[TokenWithProb]) -> None:
        if not path:
            return
        token_with_prob = path[0]
        prob = Probability(math.exp(token_with_prob.prob))

        child = self.find_child(token_with_prob.token)
        if child is None:
            child = TreeNode(token_with_prob.token, prob)
            self.children.append(child)

        child.insert_path(path[1:])

    def normalize_probs(self) -> None:
        total_prob = sum(
            child.prob for child in self.children if child.prob is not None
        )
        for child in self.children:
            if child.prob is not None:
                child.normalized_prob = Probability(child.prob / total_prob)
            child.normalize_probs()

    def path(self, tokens: Iterable[Token]) -> Iterable[TokenWithProb]:
        node = self
        for token in tokens:
            child = node.find_child(token)
            assert child
            node = child
            assert node.token and node.normalized_prob
            yield TokenWithProb(token=node.token, prob=node.normalized_prob)


class ClassifyEvaluation(BaseModel):
    """The evaluation of a single label classification run.

    Attributes:
        correct: Was the highest scoring class from the output in the set of "correct classes"
        output: The actual output from the task run
    """

    correct: bool
    output: ClassifyOutput


class AggregatedClassifyEvaluation(BaseModel):
    """The aggregated evaluation of a single label classify implementation against a dataset.

    Attributes:
        percentage_correct: Percentage of answers that were considered to be correct
        evaluation: The actual evaluations
    """

    percentage_correct: float
    evaluations: Sequence[ClassifyEvaluation]


class SingleLabelClassifyEvaluator(
    Evaluator[
        ClassifyInput,
        Sequence[str],
        ClassifyEvaluation,
        AggregatedClassifyEvaluation,
    ]
):
    def __init__(self, task: SingleLabelClassify):
        self.task = task

    def evaluate(
        self,
        input: ClassifyInput,
        logger: DebugLogger,
        expected_output: Sequence[str],
    ) -> ClassifyEvaluation:
        output = self.task.run(input, logger)
        sorted_classes = sorted(
            output.scores.items(), key=lambda item: item[1], reverse=True
        )
        if sorted_classes[0][0] in expected_output:
            correct = True
        else:
            correct = False
        return ClassifyEvaluation(correct=correct, output=output)

    def aggregate(
        self, evaluations: Sequence[ClassifyEvaluation]
    ) -> AggregatedClassifyEvaluation:
        if len(evaluations) != 0:
            correct_answers = len(
                [eval.correct for eval in evaluations if eval.correct == True]
            ) / len(evaluations)
        else:
            correct_answers = 0
        return AggregatedClassifyEvaluation(
            percentage_correct=correct_answers, evaluations=evaluations
        )
