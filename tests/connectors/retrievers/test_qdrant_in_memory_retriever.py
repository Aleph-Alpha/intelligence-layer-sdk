from collections.abc import Sequence

from pytest import fixture

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from tests.conftest import to_document


@fixture
def in_memory_retriever_documents() -> Sequence[Document]:
    return [
        Document(text="I do not like rain"),
        Document(text="Summer is warm"),
        Document(text="We are so back"),
    ]


def test_asymmetric_in_memory_retriever(
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    query = "Do you like summer?"
    documents = asymmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    assert in_memory_retriever_documents[1] == to_document(documents[0].document_chunk)
    assert len(documents) <= 2


def test_symmetric_in_memory_retriever(
    symmetric_in_memory_retriever: QdrantInMemoryRetriever,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    query = "I hate drizzle"
    documents = symmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    assert in_memory_retriever_documents[0] == to_document(documents[0].document_chunk)
    assert len(documents) <= 2
