from intelligence_layer.core import Chunk, Language, LuminousControlModel, NoOpTracer
from intelligence_layer.use_cases import (
    LongContextSummarizeInput,
    SteerableLongContextSummarize,
)
from intelligence_layer.use_cases.summarize.steerable_single_chunk_summarize import (
    SteerableSingleChunkSummarize,
)


def test_steerable_long_context_summarize_en(
    steerable_long_context_summarize: SteerableLongContextSummarize,
    long_text: str,
) -> None:
    input = LongContextSummarizeInput(text=long_text)
    output = steerable_long_context_summarize.run(input, NoOpTracer())

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


def test_steerable_long_context_summarize_adapts_to_instruction(
    luminous_control_model: LuminousControlModel,
    long_text: str,
) -> None:
    input = LongContextSummarizeInput(text=long_text)
    steerable_long_context_summarize_keyword = SteerableLongContextSummarize(
        summarize=SteerableSingleChunkSummarize(
            luminous_control_model,
            max_generated_tokens=128,
            instruction_configs={Language("en"): "Summarize using bullet points."},
        ),
        chunk=Chunk(luminous_control_model, max_tokens_per_chunk=512),
    )

    output = steerable_long_context_summarize_keyword.run(input, NoOpTracer())

    assert output.partial_summaries
    assert any(
        "bear" in partial_summary.summary
        for partial_summary in output.partial_summaries
    )
    assert all(
        partial_summary.summary.startswith("- ")
        for partial_summary in output.partial_summaries
    )
    assert len(
        " ".join(
            partial_summary.summary for partial_summary in output.partial_summaries
        )
    ) < len(long_text)
