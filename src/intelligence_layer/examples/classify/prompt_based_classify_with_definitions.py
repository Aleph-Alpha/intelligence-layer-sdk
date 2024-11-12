from collections.abc import Sequence
from math import exp

from aleph_alpha_client import Prompt
from pydantic import BaseModel

from intelligence_layer.core import (
    CompleteInput,
    CompleteOutput,
    ControlModel,
    LuminousControlModel,
    Task,
    TaskSpan,
    TextChunk,
)

from .classify import ClassifyInput, Probability, SingleLabelClassifyOutput


class LabelWithDefinition(BaseModel):
    """Defines a label with a definition.

    Attributes:
        name: Name of the label.
        definition: A definition or description of the label.
    """

    name: str
    definition: str

    def to_string(self) -> str:
        return f"{self.name}: {self.definition}"


class PromptBasedClassifyWithDefinitions(
    Task[ClassifyInput, SingleLabelClassifyOutput]
):
    INSTRUCTION: str = """Identify a class that describes the text adequately.
Reply with only the class label."""

    def __init__(
        self,
        labels_with_definitions: Sequence[LabelWithDefinition],
        model: ControlModel | None = None,
        instruction: str = INSTRUCTION,
    ) -> None:
        super().__init__()
        self._labels_with_definitions = labels_with_definitions
        self._model = model or LuminousControlModel("luminous-base-control")
        self._instruction = instruction

    def do_run(
        self, input: ClassifyInput, task_span: TaskSpan
    ) -> SingleLabelClassifyOutput:
        complete_output = self._model.complete(
            CompleteInput(
                prompt=self._get_prompt(input.chunk, input.labels),
                completion_bias_inclusion=list(input.labels),
                log_probs=len(input.labels) * 2,
            ),
            task_span,
        )
        return SingleLabelClassifyOutput(scores=self._build_scores(complete_output))

    def _get_prompt(self, chunk: TextChunk, labels: frozenset[str]) -> Prompt:
        def format_input(text: str, labels: frozenset[str]) -> str:
            definitions = "\n".join(
                label.to_string()
                for label in self._labels_with_definitions
                if label.name in labels
            )
            return f"""Labels:
{', '.join(label.name for label in self._labels_with_definitions if label.name in labels)}

Definitions:
{definitions}

Text: {text}"""

        unexpected_labels = labels - set(
            label.name for label in self._labels_with_definitions
        )
        if unexpected_labels:
            raise ValueError(f"Got unexpected labels: {', '.join(unexpected_labels)}")

        return self._model.to_instruct_prompt(
            instruction=self._instruction,
            input=format_input(text=str(chunk), labels=labels),
        )

    def _build_scores(self, complete_output: CompleteOutput) -> dict[str, Probability]:
        raw_probs: dict[str, float] = {}
        for label in self._labels_with_definitions:
            label_prob = 0.0
            assert complete_output.completions[0].log_probs
            for token, prob in complete_output.completions[0].log_probs[0].items():
                if label.name.startswith(token.strip()) and prob:
                    label_prob += exp(prob)
            raw_probs[label.name] = label_prob

        total = sum(raw_probs.values())
        return {key: Probability(value / total) for key, value in raw_probs.items()}
