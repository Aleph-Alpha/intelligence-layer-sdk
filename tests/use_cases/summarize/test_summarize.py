from pytest import fixture

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.evaluator import Example, SequenceDataset
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.summarize.long_context_high_compression_summarize import (
    LongContextHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.single_chunk_few_shot_summarize import (
    SingleChunkFewShotSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import (
    LongContextSummarizeEvaluator,
    LongContextSummarizeInput,
    SingleChunkSummarizeEvaluator,
    SingleChunkSummarizeInput,
)


@fixture
def single_chunk_summarize_evaluator(
    single_chunk_few_shot_summarize: SingleChunkFewShotSummarize,
) -> SingleChunkSummarizeEvaluator:
    return SingleChunkSummarizeEvaluator(single_chunk_few_shot_summarize)


@fixture
def long_context_summarize_evaluator(
    long_context_high_compression_summarize: LongContextHighCompressionSummarize,
) -> LongContextSummarizeEvaluator:
    return LongContextSummarizeEvaluator(long_context_high_compression_summarize)


def test_single_chunk_summarize_evaluator(
    single_chunk_summarize_evaluator: SingleChunkSummarizeEvaluator,
    chunk: Chunk,
    no_op_tracer: NoOpTracer,
) -> None:
    input = SingleChunkSummarizeInput(chunk=chunk, language=Language("en"))
    bad_expected_output = "Heute ist das Wetter schön."
    good_expected_output = (
        "The brown bear is a large mammal that lives in Eurasia and North America."
    )
    outputs = [bad_expected_output, good_expected_output]
    dataset = SequenceDataset(
        name="summarize_eval_test",
        examples=[Example(input=input, expected_output=output) for output in outputs],
    )
    aggregated_evaluation = single_chunk_summarize_evaluator.evaluate_dataset(
        dataset, no_op_tracer
    )

    assert len(aggregated_evaluation.evaluations) == len(outputs)
    assert (
        aggregated_evaluation.evaluations[0].bleu
        < aggregated_evaluation.evaluations[1].bleu
    )
    assert (
        aggregated_evaluation.evaluations[0].rouge
        < aggregated_evaluation.evaluations[1].rouge
    )


def test_long_context_summarize_evaluator(
    long_context_summarize_evaluator: LongContextSummarizeEvaluator,
    long_text: str,
    no_op_tracer: NoOpTracer,
) -> None:
    input = LongContextSummarizeInput(text=long_text, language=Language("en"))
    bad_expected_output = "Heute ist das Wetter schön."
    good_expected_output = (
        "The brown bear is a large mammal that lives in Eurasia and North America."
    )
    outputs = [bad_expected_output, good_expected_output]
    dataset = SequenceDataset(
        name="summarize_eval_test",
        examples=[Example(input=input, expected_output=output) for output in outputs],
    )
    aggregated_evaluation = long_context_summarize_evaluator.evaluate_dataset(
        dataset, no_op_tracer
    )

    assert len(aggregated_evaluation.evaluations) == len(outputs)
    assert (
        aggregated_evaluation.evaluations[0].bleu
        < aggregated_evaluation.evaluations[1].bleu
    )
    assert (
        aggregated_evaluation.evaluations[0].rouge
        < aggregated_evaluation.evaluations[1].rouge
    )
