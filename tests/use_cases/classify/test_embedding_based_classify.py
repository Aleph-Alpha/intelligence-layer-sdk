from typing import Sequence

from aleph_alpha_client import Client
from pytest import fixture, raises
from qdrant_client.http.models import models

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.core.task import Chunk
from intelligence_layer.use_cases.classify.classify import (
    ClassifyEvaluator,
    ClassifyInput,
    ClassifyOutput,
)
from intelligence_layer.use_cases.classify.embedding_based_classify import (
    LabelWithExamples,
    EmbeddingBasedClassify,
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
def embedding_based_classify(client: Client) -> EmbeddingBasedClassify:
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


def test_qdrant_search(
    qdrant_search: QdrantSearch,
    no_op_debug_logger: NoOpDebugLogger,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    search_input = QdrantSearchInput(
        query="When did Germany reunite?",
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.type",
                    match=models.MatchValue(value="doc"),
                ),
            ]
        ),
    )
    result = qdrant_search.run(search_input, no_op_debug_logger)
    assert [r.document for r in result.results] == [in_memory_retriever_documents[0]]


def test_embedding_based_classify_returns_score_for_all_labels(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )
    classify_output = embedding_based_classify.run(classify_input, NoOpDebugLogger())

    # Output contains everything we expect
    assert isinstance(classify_output, ClassifyOutput)
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_embedding_based_classify_raises_for_unknown_label(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    unknown_label = "neutral"
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative", unknown_label}),
    )
    with raises(ValueError) as e:
        embedding_based_classify.run(classify_input, NoOpDebugLogger())


def test_embedding_based_classify_works_for_empty_labels_in_request(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset(),
    )
    result = embedding_based_classify.run(classify_input, NoOpDebugLogger())
    assert result.scores == {}


def test_embedding_based_classify_works_without_examples(
    client: Client,
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
    result = embedding_based_classify.run(classify_input, NoOpDebugLogger())
    assert result.scores == {}


def test_can_evaluate_embedding_based_classify(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )
    evaluator = ClassifyEvaluator(task=embedding_based_classify)

    evaluation = evaluator.evaluate(
        input=classify_input, logger=NoOpDebugLogger(), expected_output=["positive"]
    )

    assert evaluation.correct == True
