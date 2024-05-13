from pathlib import Path
from typing import Iterable, List, Sequence
from uuid import uuid4

from pytest import fixture

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import Task, TextChunk, utc_now
from intelligence_layer.evaluation import (
    AggregationOverview,
    Aggregator,
    DatasetRepository,
    EvaluationOverview,
    Evaluator,
    Example,
    FileAggregationRepository,
    InMemoryAggregationRepository,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    Runner,
)
from intelligence_layer.examples import (
    AggregatedLabelInfo,
    AggregatedMultiLabelClassifyEvaluation,
    AggregatedSingleLabelClassifyEvaluation,
    ClassifyInput,
    EmbeddingBasedClassify,
    LabelWithExamples,
    MultiLabelClassifyAggregationLogic,
    MultiLabelClassifyEvaluation,
    MultiLabelClassifyEvaluationLogic,
    MultiLabelClassifyOutput,
)


@fixture
def embedding_based_classify(
    client: AlephAlphaClientProtocol,
) -> EmbeddingBasedClassify:
    labels_with_examples = [
        LabelWithExamples(
            name="positive",
            examples=[
                "I really like this.",
                "Wow, your hair looks great!",
                "We're so in love.",
            ],
        ),
        LabelWithExamples(
            name="negative",
            examples=[
                "I really dislike this.",
                "Ugh, Your hair looks horrible!",
                "We're not in love anymore.",
            ],
        ),
        LabelWithExamples(
            name="finance",
            examples=[
                "I have a bank meeting tomorrow.",
                "My stocks gained 100% today!",
                "The merger went through just fine.",
            ],
        ),
        LabelWithExamples(
            name="school",
            examples=[
                "Mrs. Smith was in terrible mood today.",
                "I really liked English class today",
                "The next exam period is making me anxious.",
            ],
        ),
    ]
    return EmbeddingBasedClassify(labels_with_examples, client=client)


@fixture
def embedding_based_classify_example() -> List[Example[ClassifyInput, Sequence[str]]]:
    return [
        Example(
            input=ClassifyInput(
                chunk=TextChunk("My university biology class really sucks."),
                labels=frozenset(["positive", "negative", "finance", "school"]),
            ),
            expected_output=["positive", "school"],
        )
    ]


@fixture
def embedding_based_classify_examples(
    embedding_based_classify_example: List[Example[ClassifyInput, Sequence[str]]],
) -> List[Example[ClassifyInput, Sequence[str]]]:
    return embedding_based_classify_example + [
        Example(
            input=ClassifyInput(
                chunk=TextChunk("My university banking class really sucks."),
                labels=frozenset(["positive", "negative", "finance", "school"]),
            ),
            expected_output=["negative", "finance", "school"],
        ),
        Example(
            input=ClassifyInput(
                chunk=TextChunk("I did great on the recent exam."),
                labels=frozenset(["positive", "negative", "finance", "school"]),
            ),
            expected_output=["positive", "school"],
        ),
        Example(
            input=ClassifyInput(
                chunk=TextChunk("Dogs are animals"),
                labels=frozenset(["positive", "negative", "finance", "school"]),
            ),
            expected_output=[],
        ),
    ]


@fixture
def single_entry_dataset_name(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    embedding_based_classify_example: Iterable[Example[ClassifyInput, Sequence[str]]],
) -> str:
    return in_memory_dataset_repository.create_dataset(
        examples=embedding_based_classify_example, dataset_name="test-dataset"
    ).id


@fixture
def multiple_entries_dataset_name(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    embedding_based_classify_examples: Iterable[Example[ClassifyInput, Sequence[str]]],
) -> str:
    return in_memory_dataset_repository.create_dataset(
        examples=embedding_based_classify_examples, dataset_name="test-dataset"
    ).id


@fixture
def multi_label_classify_evaluation_logic() -> MultiLabelClassifyEvaluationLogic:
    return MultiLabelClassifyEvaluationLogic()


@fixture
def multi_label_classify_aggregation_logic() -> MultiLabelClassifyAggregationLogic:
    return MultiLabelClassifyAggregationLogic()


@fixture
def classify_evaluator(
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    multi_label_classify_evaluation_logic: MultiLabelClassifyEvaluationLogic,
) -> Evaluator[
    ClassifyInput,
    MultiLabelClassifyOutput,
    Sequence[str],
    MultiLabelClassifyEvaluation,
]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "multi-label-classify",
        multi_label_classify_evaluation_logic,
    )


@fixture
def classify_aggregator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    multi_label_classify_aggregation_logic: MultiLabelClassifyAggregationLogic,
) -> Aggregator[
    MultiLabelClassifyEvaluation,
    AggregatedMultiLabelClassifyEvaluation,
]:
    return Aggregator(
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "multi-label-classify",
        multi_label_classify_aggregation_logic,
    )


@fixture
def classify_aggregator_file_repo(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    file_aggregation_repository: FileAggregationRepository,
    multi_label_classify_aggregation_logic: MultiLabelClassifyAggregationLogic,
) -> Aggregator[
    MultiLabelClassifyEvaluation,
    AggregatedMultiLabelClassifyEvaluation,
]:
    return Aggregator(
        in_memory_evaluation_repository,
        file_aggregation_repository,
        "multi-label-classify",
        multi_label_classify_aggregation_logic,
    )


@fixture
def classify_runner(
    embedding_based_classify: Task[ClassifyInput, MultiLabelClassifyOutput],
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> Runner[ClassifyInput, MultiLabelClassifyOutput]:
    return Runner(
        embedding_based_classify,
        in_memory_dataset_repository,
        in_memory_run_repository,
        "multi-label-classify",
    )


def test_multi_label_classify_evaluator_single_example(
    single_entry_dataset_name: str,
    classify_evaluator: Evaluator[
        ClassifyInput,
        MultiLabelClassifyOutput,
        Sequence[str],
        MultiLabelClassifyEvaluation,
    ],
    classify_runner: Runner[ClassifyInput, MultiLabelClassifyOutput],
) -> None:
    run_overview = classify_runner.run_dataset(single_entry_dataset_name)

    evaluation_overview = classify_evaluator.evaluate_runs(run_overview.id)
    evaluation = classify_evaluator._evaluation_repository.example_evaluations(
        evaluation_overview.id,
        MultiLabelClassifyEvaluation,
    )[0].result

    assert isinstance(evaluation, MultiLabelClassifyEvaluation)
    assert evaluation.tp == frozenset({"school"})
    assert evaluation.tn == frozenset({"finance"})
    assert evaluation.fp == frozenset({"negative"})
    assert evaluation.fn == frozenset({"positive"})


def test_multi_label_classify_evaluator_full_dataset(
    multiple_entries_dataset_name: str,
    classify_evaluator: Evaluator[
        ClassifyInput,
        MultiLabelClassifyOutput,
        Sequence[str],
        MultiLabelClassifyEvaluation,
    ],
    classify_aggregator: Aggregator[
        MultiLabelClassifyEvaluation, AggregatedMultiLabelClassifyEvaluation
    ],
    classify_runner: Runner[ClassifyInput, MultiLabelClassifyOutput],
) -> None:
    run_overview = classify_runner.run_dataset(multiple_entries_dataset_name)

    evaluation_overview = classify_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = classify_aggregator.aggregate_evaluation(
        evaluation_overview.id
    )

    assert {"positive", "negative", "finance", "school"} == set(
        aggregation_overview.statistics.class_metrics.keys()
    )


def test_confusion_matrix_in_single_label_classify_aggregation_is_compatible_with_file_repository(
    evaluation_overview: EvaluationOverview,
    tmp_path: Path,
) -> None:
    aggregated_single_label_classify_evaluation = (
        AggregatedSingleLabelClassifyEvaluation(
            percentage_correct=0.123,
            confusion_matrix={
                "happy": {"happy": 1},
                "sad": {"sad": 2},
                "angry": {"sad": 1},
            },
            by_label={
                "happy": AggregatedLabelInfo(expected_count=1, predicted_count=1),
                "sad": AggregatedLabelInfo(expected_count=3, predicted_count=2),
                "angry": AggregatedLabelInfo(expected_count=0, predicted_count=1),
            },
            missing_labels={"tired": 1},
        )
    )

    aggregation_overview = AggregationOverview(
        id=str(uuid4()),
        evaluation_overviews=frozenset([evaluation_overview]),
        start=utc_now(),
        end=utc_now(),
        successful_evaluation_count=5,
        crashed_during_evaluation_count=3,
        statistics=aggregated_single_label_classify_evaluation,
        description="dummy-aggregator",
    )

    aggregation_file_repository = FileAggregationRepository(tmp_path)
    aggregation_file_repository.store_aggregation_overview(aggregation_overview)

    aggregation_overview_from_file_repository = (
        aggregation_file_repository.aggregation_overview(
            aggregation_overview.id, AggregatedSingleLabelClassifyEvaluation
        )
    )

    assert aggregation_overview_from_file_repository == aggregation_overview
