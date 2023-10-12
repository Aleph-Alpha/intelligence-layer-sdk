from abc import abstractmethod
import math
from typing import (
    Any,
    Generic,
    Iterable,
    NewType,
    Mapping,
    Optional,
    Sequence,
)

from aleph_alpha_client import (
    Client,
    CompletionResponse,
    PromptTemplate,
    Prompt,
    TokenizationRequest,
    CompletionRequest,
)
from pydantic import BaseModel

from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from intelligence_layer.task import (
    Evaluation,
    EvaluationCase,
    Evaluator,
    Task,
    DebugLogger,
)


class Token(BaseModel):
    token: str
    token_id: int


Probability = NewType("Probability", float)
LogProb = NewType("LogProb", float)


class TokenWithProb(BaseModel):
    token: Token
    prob: Probability | LogProb


def to_aa_tokens(tokens: Sequence[Token]) -> Prompt:
    return Prompt.from_tokens([token.token_id for token in tokens])


class ClassifyInput(BaseModel):
    """Input for a classification task."""

    text: str
    """Text to be classified"""
    labels: frozenset[str]
    """Possible labels into which the text should be classified."""


class ClassifyOutput(BaseModel):
    scores: Mapping[str, Probability]
    """Mapping of the provided label (key) to corresponding score (value)

    The score represents how sure the model is that this is the correct label.
    Will be a value between 0 and 1"""


class SingleLabelClassify(Task[ClassifyInput, ClassifyOutput]):
    """Classify method for applying a single label to a given text.

    The input provides a complete set of all possible labels. The output will return a score for
    each possible label. All scores will add up to 1 and are relative to each other. The highest
    score is given to the most likely task.

    This methodology works best for classes that are easily understood, and don't require an
    explanation or examples."""

    PROMPT_TEMPLATE: str = """### Instruction:
Identify a class that describes the text adequately.
Reply with only the class label.

### Input:
{{text}}

### Response:{{label}}"""
    MODEL: str = "luminous-base-control"
    client: Client

    def __init__(self, client: Client) -> None:
        """Initializes the Task.

        Args:
        - client: the aleph alpha client
        """
        super().__init__()
        self.client = client
        self.completion_task = Completion(client)

    def run(self, input: ClassifyInput, logger: DebugLogger) -> ClassifyOutput:
        tokenized_labels = self._tokenize_labels(input.labels, logger)
        completion_responses_per_label = self._complete_per_label(
            self.MODEL, self.PROMPT_TEMPLATE, input.text, tokenized_labels, logger
        )
        log_probs_per_label = self._get_log_probs_of_labels(
            completion_responses_per_label, tokenized_labels, logger
        )
        normalized_probs_per_label = self._normalize(log_probs_per_label, logger)
        scores = self._compute_scores(normalized_probs_per_label)
        return ClassifyOutput(
            scores=scores,
        )

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

    def _complete_per_label(
        self,
        model: str,
        prompt_template_str: str,
        text: str,
        tokenized_labels: Mapping[str, Sequence[Token]],
        logger: DebugLogger,
    ) -> Mapping[str, CompletionOutput]:
        logger.log(
            "Completion",
            {
                "model": model,
                "template": prompt_template_str,
                "text": text,
            },
        )
        prompt_template = PromptTemplate(prompt_template_str)
        completion_per_label = {
            label: self._complete(
                model,
                prompt_template,
                logger.child_logger(f"Completion {label}"),
                text=text,
                label=prompt_template.embed_prompt(to_aa_tokens(tokens)),
            )
            for label, tokens in tokenized_labels.items()
        }
        return completion_per_label

    def _complete(
        self,
        model: str,
        prompt_template: PromptTemplate,
        logger: DebugLogger,
        **kwargs: Any,
    ) -> CompletionOutput:
        request = CompletionRequest(
            prompt=prompt_template.to_prompt(**kwargs),
            maximum_tokens=0,
            log_probs=0,
            tokens=True,
            echo=True,
        )
        return self.completion_task.run(
            CompletionInput(request=request, model=model), logger
        )

    def _get_log_probs_of_labels(
        self,
        completion_responses: Mapping[str, CompletionOutput],
        tokenized_labels: Mapping[str, Sequence[Token]],
        logger: DebugLogger,
    ) -> Mapping[str, Sequence[TokenWithProb]]:
        logs_probs_per_label = {
            label: self._get_log_probs_of_label(
                completion_responses[label].response, tokens
            )
            for label, tokens in tokenized_labels.items()
        }
        logger.log("Raw log probs per label", logs_probs_per_label)
        return logs_probs_per_label

    def _get_log_probs_of_label(
        self, completion_response: CompletionResponse, tokens: Sequence[Token]
    ) -> Sequence[TokenWithProb]:
        assert completion_response.completions[0].log_probs
        assert completion_response.completions[0].completion_tokens

        log_prob_dicts = completion_response.completions[0].log_probs[-len(tokens) :]
        completion_tokens = completion_response.completions[0].completion_tokens[
            -len(tokens) :
        ]
        return [
            TokenWithProb(
                token=token,
                prob=LogProb(log_prob_dict.get(completion_token, 0.0) or 0.0),
            )
            for token, log_prob_dict, completion_token in zip(
                tokens, log_prob_dicts, completion_tokens
            )
        ]

    def _tokenize_labels(
        self, labels: frozenset[str], logger: DebugLogger
    ) -> Mapping[str, Sequence[Token]]:
        tokens_per_label = {label: self._tokenize_label(label) for label in labels}
        logger.log("Tokenized Labels", tokens_per_label)
        return tokens_per_label

    def _tokenize_label(self, label: str) -> Sequence[Token]:
        """Turn a single label into list of token ids. Important so that we know how many tokens
        the label is and can retrieve the last N log probs for the label"""
        response = self.client.tokenize(
            request=TokenizationRequest(
                label + "<|endoftext|>", tokens=True, token_ids=True
            ),
            model=self.MODEL,
        )
        assert response.token_ids and response.tokens
        return [
            Token(token=token, token_id=token_id)
            for token, token_id in zip(response.tokens, response.token_ids)
        ]


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


class ClassifyEvaluationCase(EvaluationCase[ClassifyInput, Sequence[str]]):
    input: ClassifyInput
    expected_output: Sequence[str]


class ClassifyDataset(BaseModel):
    cases: Sequence[ClassifyEvaluationCase]


class SingleLabelClassifyEvaluator(Evaluator[ClassifyInput, Sequence[str]]):
    def __init__(self, task: SingleLabelClassify):
        self.task = task

    def evaluate(
        self,
        input: ClassifyInput,
        logger: DebugLogger,
        expected_output: Sequence[str],
    ) -> Evaluation:
        output = self.task.run(input, logger)
        sorted_classes = sorted(
            output.scores.items(), key=lambda item: item[1], reverse=True
        )
        if sorted_classes[0][0] in expected_output:
            correct = True
        else:
            correct = False
        return Evaluation({"correct": correct})

    # def aggregate_data(self) -> None:
    # pass

    # def evaluate_dataset(self, dataset):
    # evaluations = [self.evaluate() for data in dataset.data()]
