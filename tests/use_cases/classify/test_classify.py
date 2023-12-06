from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.evaluation.domain import Dataset, Example, SequenceDataset
from intelligence_layer.core.evaluation.repository import InMemoryEvaluationRepository
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import NoOpTracer
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
def embedding_based_classify_example() -> Example[ClassifyInput, Sequence[str]]:
    return Example(
        input=ClassifyInput(
            chunk=Chunk("My university biology class really sucks."),
            labels=frozenset(["positive", "negative", "finance", "school"]),
        ),
        expected_output=["positive", "school"],
    )


@fixture
def embedding_based_classify_dataset(
    embedding_based_classify_example: Example[ClassifyInput, Sequence[str]],
) -> Dataset[ClassifyInput, Sequence[str]]:
    return SequenceDataset(
        name="summarize_eval_test",
        examples=[
            embedding_based_classify_example,
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
        ],
    )


@fixture
def classify_evaluator(
    embedding_based_classify: Task[ClassifyInput, MultiLabelClassifyOutput],
) -> MultiLabelClassifyEvaluator:
    return MultiLabelClassifyEvaluator(
        embedding_based_classify, InMemoryEvaluationRepository()
    )


def test_multi_label_classify_evaluator_single_example(
    embedding_based_classify_example: Example[ClassifyInput, Sequence[str]],
    classify_evaluator: MultiLabelClassifyEvaluator,
    no_op_tracer: NoOpTracer,
) -> None:
    evaluation = classify_evaluator.run_and_evaluate(
        embedding_based_classify_example.input,
        embedding_based_classify_example.expected_output,
        no_op_tracer,
    )

    assert isinstance(evaluation, MultiLabelClassifyEvaluation)
    assert evaluation.tp == frozenset({"school"})
    assert evaluation.tn == frozenset({"finance"})
    assert evaluation.fp == frozenset({"negative"})
    assert evaluation.fn == frozenset({"positive"})


def test_multi_label_classify_evaluator_full_dataset(
    embedding_based_classify_dataset: Dataset[ClassifyInput, Sequence[str]],
    classify_evaluator: MultiLabelClassifyEvaluator,
) -> None:
    evaluation = classify_evaluator.evaluate_dataset(embedding_based_classify_dataset)

    assert set(["positive", "negative", "finance", "school"]) == set(
        evaluation.statistics.class_metrics.keys()
    )
