from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)


@fixture
def in_memory_retriever_documents() -> Sequence[tuple[str, Document]]:
    return [
        ("", Document(text="I do not like rain")),
        ("", Document(text="Summer is warm")),
        ("", Document(text="We are so back")),
    ]


def test_asymmetric_in_memory_retriever(
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
    in_memory_retriever_documents: Sequence[tuple[str, Document]],
) -> None:
    query = "Do you like summer?"
    documents = asymmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    assert in_memory_retriever_documents[1][1] == documents[0].document
    assert in_memory_retriever_documents[1][0] == documents[0].document_id
    assert len(documents) <= 2


def test_symmetric_in_memory_retriever(
    symmetric_in_memory_retriever: QdrantInMemoryRetriever,
    in_memory_retriever_documents: Sequence[tuple[str, Document]],
) -> None:
    query = "I hate drizzle"
    documents = symmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    assert in_memory_retriever_documents[0][1] == documents[0].document
    assert in_memory_retriever_documents[0][0] == documents[0].document_id
    assert len(documents) <= 2
