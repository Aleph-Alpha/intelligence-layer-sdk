from typing import Iterable, List, Sequence

from pytest import fixture

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import (
    Chunk,
    DatasetRepository,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    Task,
)
from intelligence_layer.core.evaluation.runner import Runner
from intelligence_layer.use_cases.classify.classify import (
    ClassifyInput,
    MultiLabelClassifyEvaluation,
    MultiLabelClassifyEvaluator,
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
    return EmbeddingBasedClassify(client, labels_with_examples)


@fixture
def embedding_based_classify_example() -> List[Example[ClassifyInput, Sequence[str]]]:
    return [
        Example(
            input=ClassifyInput(
                chunk=Chunk("My university biology class really sucks."),
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
                chunk=Chunk("My university banking class really sucks."),
                labels=frozenset(["positive", "negative", "finance", "school"]),
            ),
            expected_output=["negative", "finance", "school"],
        ),
        Example(
            input=ClassifyInput(
                chunk=Chunk("I did great on the recent exam."),
                labels=frozenset(["positive", "negative", "finance", "school"]),
            ),
            expected_output=["positive", "school"],
        ),
        Example(
            input=ClassifyInput(
                chunk=Chunk("Dogs are animals"),
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
    return in_memory_dataset_repository.create_dataset(embedding_based_classify_example)


@fixture
def multiple_entries_dataset_name(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    embedding_based_classify_examples: Iterable[Example[ClassifyInput, Sequence[str]]],
) -> str:
    return in_memory_dataset_repository.create_dataset(
        embedding_based_classify_examples
    )


@fixture
def classify_evaluator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: DatasetRepository,
) -> MultiLabelClassifyEvaluator:
    return MultiLabelClassifyEvaluator(
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "multi-label-classify",
    )


@fixture
def classify_runner(
    embedding_based_classify: Task[ClassifyInput, MultiLabelClassifyOutput],
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: DatasetRepository,
) -> Runner[ClassifyInput, MultiLabelClassifyOutput]:
    return Runner(
        embedding_based_classify,
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "multi-label-classify",
    )


def test_multi_label_classify_evaluator_single_example(
    single_entry_dataset_name: str,
    classify_evaluator: MultiLabelClassifyEvaluator,
    classify_runner: Runner[ClassifyInput, MultiLabelClassifyOutput],
) -> None:
    run_overview = classify_runner.run_dataset(single_entry_dataset_name)

    evaluation_overview = classify_evaluator.evaluate_dataset(run_overview.id)
    evaluation = classify_runner._evaluation_repository.example_evaluations(
        evaluation_overview.individual_evaluation_overviews[0].id,
        MultiLabelClassifyEvaluation,
    )[0].result

    assert isinstance(evaluation, MultiLabelClassifyEvaluation)
    assert evaluation.tp == frozenset({"school"})
    assert evaluation.tn == frozenset({"finance"})
    assert evaluation.fp == frozenset({"negative"})
    assert evaluation.fn == frozenset({"positive"})


def test_multi_label_classify_evaluator_full_dataset(
    multiple_entries_dataset_name: str,
    classify_evaluator: MultiLabelClassifyEvaluator,
    classify_runner: Runner[ClassifyInput, MultiLabelClassifyOutput],
) -> None:
    run_overview = classify_runner.run_dataset(multiple_entries_dataset_name)

    evaluation = classify_evaluator.evaluate_dataset(run_overview.id)

    assert set(["positive", "negative", "finance", "school"]) == set(
        evaluation.statistics.class_metrics.keys()
    )
