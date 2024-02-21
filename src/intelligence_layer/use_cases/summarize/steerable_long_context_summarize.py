from typing import Mapping

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import ChunkInput, ChunkTask, Task, TaskSpan
from intelligence_layer.core.chunk import ChunkOutput, ChunkOverlapTask
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.model import AlephAlphaModel, LuminousControlModel
from intelligence_layer.use_cases.summarize.steerable_single_chunk_summarize import (
    SteerableSingleChunkSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import (
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    PartialSummary,
    SingleChunkSummarizeInput,
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
        max_generated_tokens: The maximum number of tokens per sub-summary.
        max_tokens_per_chunk: The maximum number of tokens per chunk that the long text
            is divided into.
        model: A valid Aleph Alpha model.
        overlap_length_tokens: The overlap between the chunks.
        intruction_configs: Dictionary of the prompts for each language.
    """

    def __init__(
        self,
        model: AlephAlphaModel = LuminousControlModel("luminous-base-control-20240215"),
        max_generated_tokens: int = 512,
        max_tokens_per_chunk: int = 1024,
        overlap_length_tokens: int = 0,
        instruction_configs: Mapping[Language, str] = INSTRUCTION_CONFIGS,
    ) -> None:
        super().__init__()
        self._summarize = SteerableSingleChunkSummarize(
            model, max_generated_tokens, instruction_configs
        )
        self._chunk_task: Task[ChunkInput, ChunkOutput]
        if overlap_length_tokens == 0:
            self._chunk_task = ChunkTask(model, max_tokens_per_chunk)
        else:
            self._chunk_task = ChunkOverlapTask(
                model,
                max_tokens_per_chunk,
                overlap_length_tokens,
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
