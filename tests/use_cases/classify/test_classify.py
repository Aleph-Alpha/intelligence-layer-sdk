from typing import Iterable, List, Sequence

from pytest import fixture

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import Task, TextChunk
from intelligence_layer.evaluation import (
    Aggregator,
    DatasetRepository,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    Runner,
    RunRepository,
)
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    InMemoryAggregationRepository,
)
from intelligence_layer.evaluation.evaluator import Evaluator
from intelligence_layer.use_cases.classify.classify import (
    AggregatedMultiLabelClassifyEvaluation,
    ClassifyInput,
    MultiLabelClassifyAggregationLogic,
    MultiLabelClassifyEvaluation,
    MultiLabelClassifyEvaluationLogic,
    MultiLabelClassifyOutput,
)
from intelligence_layer.use_cases.classify.embedding_based_classify import (
    EmbeddingBasedClassify,
    LabelWithExamples,
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
    in_memory_run_repository: RunRepository,
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
def classify_runner(
    embedding_based_classify: Task[ClassifyInput, MultiLabelClassifyOutput],
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: RunRepository,
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
