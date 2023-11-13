from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.summarize.long_context_high_compression_summarize import (
    LongContextHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import LongContextSummarizeInput


def test_long_context_high_compression_summarize_en(
    long_context_high_compression_summarize: LongContextHighCompressionSummarize,
    long_text: str,
) -> None:
    input = LongContextSummarizeInput(text=long_text)
    output = long_context_high_compression_summarize.run(input, NoOpTracer())

    assert output.partial_summaries
    assert any(
        "bear" in partial_summary.summary
        for partial_summary in output.partial_summaries
    )
    assert len(
        " ".join(
            partial_summary.summary for partial_summary in output.partial_summaries
        )
    ) < len(long_text)
