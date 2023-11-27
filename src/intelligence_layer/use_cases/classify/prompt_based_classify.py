import math
import re
from typing import Iterable, Mapping, Optional, Sequence

from aleph_alpha_client import Prompt, PromptTemplate
from pydantic import BaseModel

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.echo import EchoInput, EchoTask, TokenWithLogProb
from intelligence_layer.core.task import Task, Token
from intelligence_layer.core.tracer import TaskSpan
from intelligence_layer.use_cases.classify.classify import (
    ClassifyInput,
    Probability,
    SingleLabelClassifyOutput,
)


class TokenWithProb(BaseModel):
    token: Token
    prob: Probability


def to_aa_tokens_prompt(tokens: Sequence[Token]) -> Prompt:
    return Prompt.from_tokens([token.token_id for token in tokens])


class PromptBasedClassify(Task[ClassifyInput, SingleLabelClassifyOutput]):
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
        >>> client = LimitedConcurrencyClient.from_token(token="AA_TOKEN")
        >>> task = PromptBasedClassify(client)
        >>> input = ClassifyInput(
                text="This is a happy text.",
                labels={"positive", "negative"}
            )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
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
    _client: AlephAlphaClientProtocol

    def __init__(self, client: AlephAlphaClientProtocol) -> None:
        super().__init__()
        self._client = client
        self._echo_task = EchoTask(client)

    def do_run(
        self, input: ClassifyInput, task_span: TaskSpan
    ) -> SingleLabelClassifyOutput:
        log_probs_per_label = self._log_probs_per_label(
            text_to_classify=input.chunk,
            labels=input.labels,
            model=self.MODEL,
            task_span=task_span,
        )
        task_span.log("Log probs per label", log_probs_per_label)
        normalized_probs_per_label = self._normalize(log_probs_per_label, task_span)
        scores = self._compute_scores(normalized_probs_per_label)
        return SingleLabelClassifyOutput(
            scores=scores,
        )

    def _log_probs_per_label(
        self,
        text_to_classify: str,
        labels: frozenset[str],
        model: str,
        task_span: TaskSpan,
    ) -> Mapping[str, Sequence[TokenWithLogProb]]:
        prompt = PromptTemplate(template_str=self.PROMPT_TEMPLATE).to_prompt(
            text=text_to_classify
        )
        inputs = (
            EchoInput(
                prompt=prompt,
                expected_completion=self._prepare_label_for_echo_task(label),
                model=model,
            )
            for label in labels
        )
        outputs = self._echo_task.run_concurrently(inputs, task_span)
        return {
            label: output.tokens_with_log_probs
            for label, output in zip(labels, outputs)
        }

    def _prepare_label_for_echo_task(self, label: str) -> str:
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
        log_probs_per_label: Mapping[str, Sequence[TokenWithLogProb]],
        task_span: TaskSpan,
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
        task_span.log("Normalized Probs", normalized_probs)
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

    def insert_path(self, path: Sequence[TokenWithLogProb]) -> None:
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
