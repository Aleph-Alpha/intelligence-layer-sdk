from pytest import fixture

from intelligence_layer.core import (
    Example,
    InMemoryEvaluationRepository,
    SequenceDataset,
)
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language
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
    SummarizeEvaluation,
)


@fixture
def single_chunk_summarize_evaluator(
    single_chunk_few_shot_summarize: SingleChunkFewShotSummarize,
) -> SingleChunkSummarizeEvaluator:
    return SingleChunkSummarizeEvaluator(
        single_chunk_few_shot_summarize, InMemoryEvaluationRepository()
    )


@fixture
def long_context_summarize_evaluator(
    long_context_high_compression_summarize: LongContextHighCompressionSummarize,
) -> LongContextSummarizeEvaluator:
    return LongContextSummarizeEvaluator(
        long_context_high_compression_summarize, InMemoryEvaluationRepository()
    )


def test_single_chunk_summarize_evaluator(
    single_chunk_summarize_evaluator: SingleChunkSummarizeEvaluator,
    chunk: Chunk,
    no_op_tracer: NoOpTracer,
) -> None:
    input = SingleChunkSummarizeInput(chunk=chunk, language=Language("en"))
    bad_example = Example(
        input=input, expected_output="Heute ist das Wetter schön.", id="bad"
    )
    good_example = Example(
        input=input,
        expected_output="The brown bear is a large mammal that lives in Eurasia and North America.",
        id="good",
    )
    dataset = SequenceDataset(
        name="summarize_eval_test",
        examples=[good_example, bad_example],
    )
    evaluation_overview = single_chunk_summarize_evaluator.evaluate_dataset(
        dataset, no_op_tracer
    )

    assert len(evaluation_overview.statistics.evaluations) == len(dataset.examples)
    good_result = (
        single_chunk_summarize_evaluator._evaluation_repository.example_evaluation(
            evaluation_overview.id,
            "good",
            SummarizeEvaluation,
        )
    )
    bad_result = (
        single_chunk_summarize_evaluator._evaluation_repository.example_evaluation(
            evaluation_overview.id,
            "bad",
            SummarizeEvaluation,
        )
    )
    assert bad_result and isinstance(bad_result.result, SummarizeEvaluation)
    assert good_result and isinstance(good_result.result, SummarizeEvaluation)
    assert bad_result.result.bleu < good_result.result.bleu
    assert bad_result.result.rouge < good_result.result.rouge


def test_long_context_summarize_evaluator(
    long_context_summarize_evaluator: LongContextSummarizeEvaluator,
    long_text: str,
    no_op_tracer: NoOpTracer,
) -> None:
    input = LongContextSummarizeInput(text=long_text, language=Language("en"))
    bad_example = Example(
        input=input, expected_output="Heute ist das Wetter schön.", id="bad"
    )
    good_example = Example(
        input=input,
        expected_output="The brown bear is a large mammal that lives in Eurasia and North America.",
        id="good",
    )
    dataset = SequenceDataset(
        name="summarize_eval_test",
        examples=[good_example, bad_example],
    )
    evaluation_overview = long_context_summarize_evaluator.evaluate_dataset(
        dataset, no_op_tracer
    )

    assert len(evaluation_overview.statistics.evaluations) == len(dataset.examples)
    good_result = (
        long_context_summarize_evaluator._evaluation_repository.example_evaluation(
            evaluation_overview.id,
            "good",
            SummarizeEvaluation,
        )
    )
    bad_result = (
        long_context_summarize_evaluator._evaluation_repository.example_evaluation(
            evaluation_overview.id,
            "bad",
            SummarizeEvaluation,
        )
    )
    assert bad_result and isinstance(bad_result.result, SummarizeEvaluation)
    assert good_result and isinstance(good_result.result, SummarizeEvaluation)
    assert bad_result.result.bleu < good_result.result.bleu
    assert bad_result.result.rouge < good_result.result.rouge
