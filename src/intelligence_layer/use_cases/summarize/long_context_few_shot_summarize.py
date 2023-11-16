from typing import Mapping

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.chunk import ChunkInput, ChunkTask
from intelligence_layer.core.complete import FewShotConfig
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan
from intelligence_layer.use_cases.summarize.single_chunk_few_shot_summarize import (
    SingleChunkFewShotSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import (
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    PartialSummary,
    SingleChunkSummarizeInput,
)


class LongContextFewShotSummarize(
    Task[LongContextSummarizeInput, LongContextSummarizeOutput]
):
    """Condenses a long text into a summary.

    Generate a summary given a few-shot setup.

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
        few_shot_configs: Mapping[Language, FewShotConfig],
        model: str,
        max_generated_tokens: int,
        max_tokens_per_chunk: int,
    ) -> None:
        self._single_chunk_summarize = SingleChunkFewShotSummarize(
            client, model, max_generated_tokens, few_shot_configs
        )
        self._chunk = ChunkTask(client, model, max_tokens_per_chunk)

    def do_run(
        self, input: LongContextSummarizeInput, task_span: TaskSpan
    ) -> LongContextSummarizeOutput:
        chunk_output = self._chunk.run(ChunkInput(text=input.text), task_span)
        summary_outputs = self._single_chunk_summarize.run_concurrently(
            [
                SingleChunkSummarizeInput(chunk=c, language=input.language)
                for c in chunk_output.chunks
            ],
            task_span,
        )
        return LongContextSummarizeOutput(
            partial_summaries=[
                PartialSummary(summary=summary_output.summary, chunk=chunk)
                for summary_output, chunk in zip(summary_outputs, chunk_output.chunks)
            ]
        )
