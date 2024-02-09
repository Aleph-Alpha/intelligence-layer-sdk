from pytest import fixture

from intelligence_layer.core import (
    Chunk,
    DatasetRepository,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    Language,
    NoOpTracer,
)
from intelligence_layer.core.evaluation.evaluator import EvaluationRepository
from intelligence_layer.core.evaluation.runner import Runner
from intelligence_layer.use_cases.summarize.long_context_high_compression_summarize import (
    LongContextHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.single_chunk_few_shot_summarize import (
    SingleChunkFewShotSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import (
    LongContextSummarizeEvaluator,
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    SingleChunkSummarizeEvaluator,
    SingleChunkSummarizeInput,
    SummarizeEvaluation,
    SummarizeOutput,
)


@fixture
def single_chunk_summarize_evaluator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> SingleChunkSummarizeEvaluator:
    return SingleChunkSummarizeEvaluator(
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "single-chunk-summarize",
    )


@fixture
def single_chunk_summarize_runner(
    single_chunk_few_shot_summarize: SingleChunkFewShotSummarize,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> Runner[SingleChunkSummarizeInput, SummarizeOutput]:
    return Runner(
        single_chunk_few_shot_summarize,
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "single-chunk-summarize",
    )


@fixture
def long_context_summarize_evaluator(
    in_memory_evaluation_repository: EvaluationRepository,
    in_memory_dataset_repository: DatasetRepository,
) -> LongContextSummarizeEvaluator:
    return LongContextSummarizeEvaluator(
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "long-context-summarize",
    )


@fixture
def long_context_summarize_runner(
    long_context_high_compression_summarize: LongContextHighCompressionSummarize,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: DatasetRepository,
) -> Runner[LongContextSummarizeInput, LongContextSummarizeOutput]:
    return Runner(
        long_context_high_compression_summarize,
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "long-context-summarize",
    )


def test_single_chunk_summarize_evaluator(
    single_chunk_summarize_evaluator: SingleChunkSummarizeEvaluator,
    single_chunk_summarize_runner: Runner[str, str],
    chunk: Chunk,
    no_op_tracer: NoOpTracer,
    in_memory_dataset_repository: InMemoryDatasetRepository,
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
    dataset_name = in_memory_dataset_repository.create_dataset(
        [good_example, bad_example]
    )
    run_overview = single_chunk_summarize_runner.run_dataset(dataset_name)

    evaluation_overview = single_chunk_summarize_evaluator.evaluate_dataset(
        run_overview.id
    )

    assert evaluation_overview.successful_count == 2
    individual_evaluation_id = next(
        iter(evaluation_overview.individual_evaluation_overviews)
    ).id
    good_result = (
        single_chunk_summarize_evaluator._evaluation_repository.example_evaluation(
            individual_evaluation_id,
            "good",
            SummarizeEvaluation,
        )
    )
    bad_result = (
        single_chunk_summarize_evaluator._evaluation_repository.example_evaluation(
            individual_evaluation_id,
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
    long_context_summarize_runner: Runner[str, str],
    in_memory_dataset_repository: InMemoryDatasetRepository,
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
    dataset_name = in_memory_dataset_repository.create_dataset(
        [good_example, bad_example]
    )
    run_overview = long_context_summarize_runner.run_dataset(dataset_name)

    evaluation_overview = long_context_summarize_evaluator.evaluate_dataset(
        run_overview.id
    )

    assert evaluation_overview.successful_count == 2
    individual_evaluation_id = next(
        iter(evaluation_overview.individual_evaluation_overviews)
    ).id
    good_result = (
        long_context_summarize_evaluator._evaluation_repository.example_evaluation(
            individual_evaluation_id,
            "good",
            SummarizeEvaluation,
        )
    )
    bad_result = (
        long_context_summarize_evaluator._evaluation_repository.example_evaluation(
            individual_evaluation_id,
            "bad",
            SummarizeEvaluation,
        )
    )
    assert bad_result and isinstance(bad_result.result, SummarizeEvaluation)
    assert good_result and isinstance(good_result.result, SummarizeEvaluation)
    assert bad_result.result.bleu < good_result.result.bleu
    assert bad_result.result.rouge < good_result.result.rouge
