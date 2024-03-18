from pytest import fixture

from intelligence_layer.core import Language, NoOpTracer, TextChunk
from intelligence_layer.evaluation import (
    Aggregator,
    DatasetRepository,
    EvaluationRepository,
    Evaluator,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    Runner,
    RunRepository,
)
from intelligence_layer.evaluation.aggregation.aggregation_repository import (
    InMemoryAggregationRepository,
)
from intelligence_layer.use_cases import (
    AggregatedSummarizeEvaluation,
    LongContextSummarizeAggregationLogic,
    LongContextSummarizeEvaluationLogic,
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    SingleChunkSummarizeAggregationLogic,
    SingleChunkSummarizeEvaluationLogic,
    SingleChunkSummarizeInput,
    SteerableLongContextSummarize,
    SteerableSingleChunkSummarize,
    SummarizeEvaluation,
    SummarizeOutput,
)


@fixture
def single_chunk_summarize_aggregation_logic() -> SingleChunkSummarizeAggregationLogic:
    return SingleChunkSummarizeAggregationLogic()


@fixture
def single_chunk_summarize_eval_logic() -> SingleChunkSummarizeEvaluationLogic:
    return SingleChunkSummarizeEvaluationLogic()


@fixture
def single_chunk_summarize_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    single_chunk_summarize_eval_logic: SingleChunkSummarizeEvaluationLogic,
) -> Evaluator[
    SingleChunkSummarizeInput,
    SummarizeOutput,
    str,
    SummarizeEvaluation,
]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "single-chunk-summarize",
        single_chunk_summarize_eval_logic,
    )


@fixture
def single_chunk_summarize_aggregator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    single_chunk_summarize_aggregation_logic: SingleChunkSummarizeAggregationLogic,
) -> Aggregator[SummarizeEvaluation, AggregatedSummarizeEvaluation]:
    return Aggregator(
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "single-chunk-summarize",
        single_chunk_summarize_aggregation_logic,
    )


@fixture
def single_chunk_summarize_runner(
    steerable_single_chunk_summarize: SteerableSingleChunkSummarize,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> Runner[SingleChunkSummarizeInput, SummarizeOutput]:
    return Runner(
        steerable_single_chunk_summarize,
        in_memory_dataset_repository,
        in_memory_run_repository,
        "single-chunk-summarize",
    )


@fixture
def long_context_summarize_evaluation_logic() -> LongContextSummarizeEvaluationLogic:
    return LongContextSummarizeEvaluationLogic()


@fixture
def long_context_summarize_aggregation_logic() -> LongContextSummarizeAggregationLogic:
    return LongContextSummarizeAggregationLogic()


@fixture
def long_context_summarize_evaluator(
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: RunRepository,
    in_memory_evaluation_repository: EvaluationRepository,
    long_context_summarize_evaluation_logic: LongContextSummarizeEvaluationLogic,
) -> Evaluator[
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    str,
    SummarizeEvaluation,
]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "long-context-summarize",
        long_context_summarize_evaluation_logic,
    )


@fixture
def long_context_summarize_aggregator(
    in_memory_evaluation_repository: EvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    long_context_summarize_aggregation_logic: LongContextSummarizeAggregationLogic,
) -> Aggregator[
    SummarizeEvaluation,
    AggregatedSummarizeEvaluation,
]:
    return Aggregator(
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "long-context-summarize",
        long_context_summarize_aggregation_logic,
    )


@fixture
def long_context_summarize_runner(
    steerable_long_context_summarize: SteerableLongContextSummarize,
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> Runner[LongContextSummarizeInput, LongContextSummarizeOutput]:
    return Runner(
        steerable_long_context_summarize,
        in_memory_dataset_repository,
        in_memory_run_repository,
        "long-context-summarize",
    )


def test_single_chunk_summarize_evaluator(
    single_chunk_summarize_evaluator: Evaluator[
        SingleChunkSummarizeInput,
        SummarizeOutput,
        str,
        SummarizeEvaluation,
    ],
    single_chunk_summarize_aggregator: Aggregator[
        SummarizeEvaluation,
        AggregatedSummarizeEvaluation,
    ],
    single_chunk_summarize_runner: Runner[str, str],
    chunk: TextChunk,
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
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[good_example, bad_example], dataset_name="test-dataset"
    ).id
    run_overview = single_chunk_summarize_runner.run_dataset(dataset_id)

    evaluation_overview = single_chunk_summarize_evaluator.evaluate_runs(
        run_overview.id
    )
    aggregation_overview = single_chunk_summarize_aggregator.aggregate_evaluation(
        evaluation_overview.id
    )

    assert aggregation_overview.successful_evaluation_count == 2
    individual_evaluation_id = next(iter(aggregation_overview.evaluation_overviews)).id
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
    long_context_summarize_evaluator: Evaluator[
        LongContextSummarizeInput,
        LongContextSummarizeOutput,
        str,
        SummarizeEvaluation,
    ],
    long_context_summarize_aggregator: Aggregator[
        SummarizeEvaluation,
        AggregatedSummarizeEvaluation,
    ],
    long_context_summarize_runner: Runner[str, str],
    in_memory_dataset_repository: InMemoryDatasetRepository,
    long_text: str,
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
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[good_example, bad_example], dataset_name="test-dataset"
    ).id
    run_overview = long_context_summarize_runner.run_dataset(dataset_id)

    evaluation_overview = long_context_summarize_evaluator.evaluate_runs(
        run_overview.id
    )
    aggregation_overview = long_context_summarize_aggregator.aggregate_evaluation(
        evaluation_overview.id
    )

    assert aggregation_overview.successful_evaluation_count == 2
    individual_evaluation_id = next(iter(aggregation_overview.evaluation_overviews)).id
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
