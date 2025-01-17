import math
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

from aleph_alpha_client import Tokens
from pydantic import BaseModel

from intelligence_layer.core import (
    Echo,
    EchoInput,
    EchoOutput,
    LuminousControlModel,
    RichPrompt,
    Task,
    TaskSpan,
    Token,
    TokenWithLogProb,
)
from intelligence_layer.core.model import ControlModel
from intelligence_layer.examples.classify.classify import (
    ClassifyInput,
    Probability,
    SingleLabelClassifyOutput,
)


class TokenWithProb(BaseModel):
    token: Token
    prob: Probability


def to_aa_tokens_prompt(tokens: Sequence[Token]) -> RichPrompt:
    return RichPrompt([Tokens([token.token_id for token in tokens], [])])


class PromptBasedClassify(Task[ClassifyInput, SingleLabelClassifyOutput]):
    """Task that classifies a given input text with one of the given classes.

    The input contains a complete set of all possible labels. The output will return a score for
    each possible label. All scores will add up to 1 and are relative to each other. The highest
    score is given to the most likely class.

    This methodology works best for classes that are easily understood, and don't require an
    explanation or examples.

    Args:
        model: The model used throughout the task for model related API calls. Defaults
            to luminous-base-control.
        echo: echo-task used to compute the score for each label. Defaults to :class:`Echo`.
        instruction: The prompt to use. Check the class for the default.

    Attributes:
        PROMPT_TEMPLATE_STR: The prompt template used for answering the question.
            'text' and 'labels' will be inserted here.
        MODEL: A valid Aleph Alpha model name.

    Example:
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.core import TextChunk
        >>> from intelligence_layer.examples import ClassifyInput
        >>> from intelligence_layer.examples import PromptBasedClassify


        >>> task = PromptBasedClassify()
        >>> input = ClassifyInput(
        ...     chunk=TextChunk("This is a happy text."), labels=frozenset({"positive", "negative"})
        ... )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    INSTRUCTION: str = """Identify a class that describes the text adequately.
Reply with only the class label."""

    def __init__(
        self,
        model: ControlModel | None = None,
        echo: Task[EchoInput, EchoOutput] | None = None,
        instruction: str = INSTRUCTION,
    ) -> None:
        super().__init__()
        self._model = model or LuminousControlModel("luminous-base-control")
        if not isinstance(self._model, LuminousControlModel):
            warnings.warn(
                "PromptBasedClassify was build for luminous models. LLama models may not work correctly. "
                "Proceed with caution and testing.",
                UserWarning,
            )
        self._echo_task = echo or Echo(self._model)
        self.instruction = instruction

    def do_run(
        self, input: ClassifyInput, task_span: TaskSpan
    ) -> SingleLabelClassifyOutput:
        log_probs_per_label = self._log_probs_per_label(
            text_to_classify=input.chunk,
            labels=input.labels,
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
        task_span: TaskSpan,
    ) -> Mapping[str, Sequence[TokenWithLogProb]]:
        prompt = self._model.to_instruct_prompt(
            instruction=self.instruction, input=text_to_classify
        )
        inputs = (
            EchoInput(
                prompt=prompt,
                expected_completion=self._prepare_label_for_echo_task(label),
            )
            for label in labels
        )
        outputs = self._echo_task.run_concurrently(inputs, task_span)
        return {
            label: output.tokens_with_log_probs
            for label, output in zip(labels, outputs, strict=True)
        }

    def _prepare_label_for_echo_task(self, label: str) -> str:
        label = label if re.match(r"^\s+", label) else f" {label}"
        return label + self._model.eot_token

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
        """Inserts a path into the tree without changing the original probability.

        Args:
            path: Path to insert

        Temporarily here until we change this data structure to be more versatile
        """
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
