from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.summarize.single_chunk_few_shot_summarize import (
    SingleChunkFewShotSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import SingleChunkSummarizeInput


def test_high_compression_summarize_en(
    single_chunk_few_shot_summarize: SingleChunkFewShotSummarize, chunk: Chunk
) -> None:
    input = SingleChunkSummarizeInput(chunk=chunk, language=Language("en"))
    output = single_chunk_few_shot_summarize.run(input, NoOpTracer())

    assert output.summary
    assert "bear" in output.summary.lower()
    assert len(output.summary) < len(chunk)


def test_high_compression_summarize_is_language_sensitive(
    single_chunk_few_shot_summarize: SingleChunkFewShotSummarize, chunk: Chunk
) -> None:
    input_en = SingleChunkSummarizeInput(chunk=chunk, language=Language("en"))
    input_de = SingleChunkSummarizeInput(chunk=chunk, language=Language("de"))
    output_en, output_de = single_chunk_few_shot_summarize.run_concurrently(
        [input_en, input_de], NoOpTracer(), concurrency_limit=2
    )

    assert output_en.summary != output_de.summary
