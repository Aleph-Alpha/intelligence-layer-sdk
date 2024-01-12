from typing import Mapping

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import ChunkInput, ChunkTask, Task, TaskSpan
from intelligence_layer.core.detect_language import Language
from intelligence_layer.use_cases import (
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    PartialSummary,
    SingleChunkSummarizeInput,
    SteerableSingleChunkSummarize
)

INSTRUCTION_CONFIGS = {
    Language("en"): "Summarize the text in a single paragraph.",
    Language("de"): "Fasse den Text in einem Paragraphen zusammen.",
}


class SteerableLongContextSummarize(
    Task[LongContextSummarizeInput, LongContextSummarizeOutput]
):
    """Condenses a long text into a summary.

    Generate a summary given an instruction setup.

    Note:
        - `model` provided should be a vanilla model, such as "luminous-base".

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        few_shot_configs: A mapping of valid `Language` to `FewShotConfig` for each
            supported language.
        model: A valid Aleph Alpha model name.
        max_generated_tokens: The maximum number of tokens per sub-summary.
        max_tokens_per_chunk: The maximum number of tokens per chunk that the long text
            is divided into.
        allowed_languages: List of languages to which the language detection is limited (ISO619).
        fallback_language: The default language of the output.
    """

    def __init__(
        self,
        client: AlephAlphaClientProtocol,
        max_generated_tokens: int,
        max_tokens_per_chunk: int,
        model: str = "luminous-base-control",
        instruction_configs: Mapping[Language, str] = INSTRUCTION_CONFIGS,
    ) -> None:
        super().__init__()
        self._summarize = SteerableSingleChunkSummarize(
            client, model, max_generated_tokens, instruction_configs
        )
        self._chunk_task = ChunkTask(
            client, model=model, max_tokens_per_chunk=max_tokens_per_chunk
        )

    def do_run(
        self, input: LongContextSummarizeInput, task_span: TaskSpan
    ) -> LongContextSummarizeOutput:
        chunk_output = self._chunk_task.run(ChunkInput(text=input.text), task_span)
        summary_outputs = self._summarize.run_concurrently(
            [
                SingleChunkSummarizeInput(chunk=chunk, language=input.language)
                for chunk in chunk_output.chunks
            ],
            task_span,
        )
        return LongContextSummarizeOutput(
            partial_summaries=[
                PartialSummary(
                    summary=summary_output.summary,
                    chunk=chunk,
                    generated_tokens=summary_output.generated_tokens,
                )
                for summary_output, chunk in zip(summary_outputs, chunk_output.chunks)
            ]
        )
