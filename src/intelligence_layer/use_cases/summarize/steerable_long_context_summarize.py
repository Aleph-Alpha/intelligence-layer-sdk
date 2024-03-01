from intelligence_layer.core import (
    Chunk,
    ChunkInput,
    ChunkOutput,
    ControlModel,
    LuminousControlModel,
    Task,
    TaskSpan,
)
from intelligence_layer.use_cases.summarize.steerable_single_chunk_summarize import (
    SteerableSingleChunkSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import (
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    PartialSummary,
    SingleChunkSummarizeInput,
    SummarizeOutput,
)


class SteerableLongContextSummarize(
    Task[LongContextSummarizeInput, LongContextSummarizeOutput]
):
    """Condenses a long text into a summary.

    Generate a summary given an instruction setup.

    Args:
        summarize: The summarize task that is used to summarize a single chunk.
            Make sure that this and the chunk task use the same model.
            Defaults to :class:`SteerableSingleChunkSummarize` .
        chunk: The chunk task that is used to chunk the long text into smaller pieces
            such that a single chunk fits into the context of the model.
            Make sure that this and the summarize task use the same model.
            Defaults to :class:`Chunk` .
        model: A valid Aleph Alpha control model. This is passed on to the
            default summarize and chunk tasks. So it is ignored when the
            defaults for both tasks are overwritten.
            Defaults to luminous-base-control-20240215.
    """

    def __init__(
        self,
        summarize: Task[SingleChunkSummarizeInput, SummarizeOutput] | None = None,
        chunk: Task[ChunkInput, ChunkOutput] | None = None,
        model: ControlModel | None = None,
    ) -> None:
        super().__init__()
        model = model or LuminousControlModel("luminous-base-control-20240215")
        self._summarize = summarize or SteerableSingleChunkSummarize(
            model, max_generated_tokens=512
        )
        self._chunk_task = chunk or Chunk(model, max_tokens_per_chunk=1024)

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
