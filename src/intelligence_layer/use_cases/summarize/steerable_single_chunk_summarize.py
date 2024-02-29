from typing import Mapping

from intelligence_layer.core import (
    CompleteInput,
    ControlModel,
    Language,
    LuminousControlModel,
    Task,
    TaskSpan,
)
from intelligence_layer.use_cases.summarize.summarize import (
    SingleChunkSummarizeInput,
    SummarizeOutput,
)

INSTRUCTION_CONFIGS = {
    Language("en"): "Summarize the text in a single paragraph.",
    Language("de"): "Fasse den Text in einem Paragraphen zusammen.",
}


class SteerableSingleChunkSummarize(Task[SingleChunkSummarizeInput, SummarizeOutput]):
    """Summarises a text given an instruction.

    Args:
        model: A valid Aleph Alpha model.
        maximum_tokens: The maximum number of tokens to be generated.
        instruction_configs: A mapping of valid `Language` to `str` for each
            supported language.
    """

    def __init__(
        self,
        model: ControlModel | None = None,
        max_generated_tokens: int = 256,
        instruction_configs: Mapping[Language, str] = INSTRUCTION_CONFIGS,
    ) -> None:
        self._model = model or LuminousControlModel("luminous-base-control-20240215")
        self._max_generated_tokens = max_generated_tokens
        self._instruction_configs = instruction_configs

    def do_run(
        self, input: SingleChunkSummarizeInput, task_span: TaskSpan
    ) -> SummarizeOutput:
        instruction = self._instruction_configs.get(input.language)
        if not instruction:
            raise ValueError(f"Could not find `prompt_config` for {input.language}.")
        completion = self._model.complete(
            CompleteInput(
                prompt=self._model.to_instruct_prompt(instruction, input.chunk),
                maximum_tokens=self._max_generated_tokens,
            ),
            task_span,
        )
        return SummarizeOutput(
            summary=completion.completion.strip(),
            generated_tokens=completion.generated_tokens,
        )
