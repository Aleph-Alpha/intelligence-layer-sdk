from typing import Sequence

from pytest import fixture, raises
from qdrant_client.http.models import models

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.evaluator import InMemoryEvaluationRepository
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.classify.classify import (
    ClassifyEvaluation,
    ClassifyInput,
    MultiLabelClassifyEvaluator,
    MultiLabelClassifyOutput,
)
from intelligence_layer.use_cases.classify.embedding_based_classify import (
    EmbeddingBasedClassify,
    LabelWithExamples,
    QdrantSearch,
    QdrantSearchInput,
)


@fixture
def in_memory_retriever_documents() -> Sequence[Document]:
    return [
        Document(
            text="Germany reunited. I kind of fit and am of the correct type.",
            metadata={"type": "doc"},
        ),
        Document(
            text="Cats are small animals. Well, I do not fit at all and I am of the correct type.",
            metadata={"type": "no doc"},
        ),
        Document(
            text="Germany reunited in 1990. This document fits perfectly but it is of the wrong type.",
            metadata={"type": "no doc"},
        ),
    ]


@fixture
def qdrant_search(
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
) -> QdrantSearch:
    return QdrantSearch(asymmetric_in_memory_retriever)


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
    ]
    return EmbeddingBasedClassify(labels_with_examples, client)


@fixture
def classify_evaluator(
    embedding_based_classify: Task[ClassifyInput, MultiLabelClassifyOutput],
) -> MultiLabelClassifyEvaluator:
    return MultiLabelClassifyEvaluator(
        embedding_based_classify, InMemoryEvaluationRepository()
    )


def test_qdrant_search(
    qdrant_search: QdrantSearch,
    no_op_tracer: NoOpTracer,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    search_input = QdrantSearchInput(
        query="When did Germany reunite?",
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="doc"),
                ),
            ]
        ),
    )
    result = qdrant_search.run(search_input, no_op_tracer)
    assert [r.document for r in result.results] == [in_memory_retriever_documents[0]]


def test_embedding_based_classify_returns_score_for_all_labels(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )
    classify_output = embedding_based_classify.run(classify_input, NoOpTracer())

    # Output contains everything we expect
    assert isinstance(classify_output, MultiLabelClassifyOutput)
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_embedding_based_classify_raises_for_unknown_label(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    unknown_label = "neutral"
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative", unknown_label}),
    )
    with raises(ValueError) as _:
        embedding_based_classify.run(classify_input, NoOpTracer())


def test_embedding_based_classify_works_for_empty_labels_in_request(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset(),
    )
    result = embedding_based_classify.run(classify_input, NoOpTracer())
    assert result.scores == {}


def test_embedding_based_classify_works_without_examples(
    client: AlephAlphaClientProtocol,
) -> None:
    labels_with_examples = [
        LabelWithExamples(
            name="positive",
            examples=[],
        ),
        LabelWithExamples(
            name="negative",
            examples=[],
        ),
    ]
    embedding_based_classify = EmbeddingBasedClassify(labels_with_examples, client)
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset(),
    )
    result = embedding_based_classify.run(classify_input, NoOpTracer())
    assert result.scores == {}


def test_can_evaluate_embedding_based_classify(
    classify_evaluator: MultiLabelClassifyEvaluator,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )

    evaluation = classify_evaluator.evaluate(
        input=classify_input,
        tracer=NoOpTracer(),
        expected_output=["positive"],
    )

    assert isinstance(evaluation, ClassifyEvaluation)
    assert evaluation.correct is True
