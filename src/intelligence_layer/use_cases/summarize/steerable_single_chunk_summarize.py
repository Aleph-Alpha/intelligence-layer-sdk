from typing import Mapping

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core import Language, Task, TaskSpan
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
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.
        maximum_tokens: The maximum number of tokens to be generated.
        instruction_configs: A mapping of valid `Language` to `str` for each
            supported language.
    """

    def __init__(
        self,
        client: AlephAlphaClientProtocol,
        model: str,
        maximum_tokens: int,
        instruction_configs: Mapping[Language, str] = INSTRUCTION_CONFIGS,
    ) -> None:
        self._instruction_configs = instruction_configs
        self._instruct = Instruct(client, model)
        self._model = model
        self._maximum_tokens = maximum_tokens

    def do_run(
        self, input: SingleChunkSummarizeInput, task_span: TaskSpan
    ) -> SummarizeOutput:
        prompt_output = self._get_prompt_and_complete(input, task_span)
        return SummarizeOutput(
            summary=prompt_output.completion.strip(),
            generated_tokens=prompt_output.generated_tokens,
        )

    def _get_prompt_and_complete(
        self, input: SingleChunkSummarizeInput, task_span: TaskSpan
    ) -> PromptOutput:
        instruction = self._instruction_configs.get(input.language)
        if not instruction:
            raise ValueError(f"Could not find `prompt_config` for {input.language}.")
        return self._instruct.run(
            InstructInput(
                instruction=instruction,
                input=input.chunk,
                maximum_response_tokens=self._maximum_tokens,
            ),
            task_span,
        )
