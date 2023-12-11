from pathlib import Path
from pytest import fixture
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.summarize.long_context_high_compression_summarize import (
    LongContextHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.recursive_summarize import (
    RecursiveSummarize,
    RecursiveSummarizeInput,
)


@fixture
def very_long_text() -> str:
    with open(
        Path(__file__).parent / "very_long_text.txt", "r", encoding="utf-8"
    ) as file:
        return file.read()


def test_recursive_summarize(
    very_long_text: str,
    long_context_high_compression_summarize: LongContextHighCompressionSummarize,
) -> None:
    input = RecursiveSummarizeInput(text=very_long_text, n_loops=3)
    task = RecursiveSummarize(long_context_high_compression_summarize)
    output = task.run(input, NoOpTracer())

    assert len(output.summary) < len(very_long_text) / 100
    assert "new orleans" in output.summary.lower()
