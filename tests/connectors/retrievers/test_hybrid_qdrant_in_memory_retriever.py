from collections.abc import Sequence

from pytest import fixture

from intelligence_layer.connectors import AlephAlphaClientProtocol, RetrieverType
from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.hybrid_qdrant_in_memory_retriever import (
    HybridQdrantInMemoryRetriever,
)
from tests.conftest import to_document


@fixture
def in_memory_retriever_documents() -> Sequence[Document]:
    return [
        Document(text="Summer is warm but I like it"),
        Document(text="I do not like rain"),
        Document(text="We are so back"),
        Document(text="Summer rain is rejuvenating"),
    ]


@fixture
def hybrid_asymmetric_in_memory_retriever(
    client: AlephAlphaClientProtocol,
    in_memory_retriever_documents: Sequence[Document],
) -> HybridQdrantInMemoryRetriever:
    return HybridQdrantInMemoryRetriever(
        in_memory_retriever_documents,
        client=client,
        k=2,
        retriever_type=RetrieverType.ASYMMETRIC,
    )


@fixture
def hybrid_symmetric_in_memory_retriever(
    client: AlephAlphaClientProtocol,
    in_memory_retriever_documents: Sequence[Document],
) -> HybridQdrantInMemoryRetriever:
    return HybridQdrantInMemoryRetriever(
        in_memory_retriever_documents,
        client=client,
        k=2,
        retriever_type=RetrieverType.SYMMETRIC,
    )


def test_asymmetric_in_memory_retriever(
    hybrid_asymmetric_in_memory_retriever: HybridQdrantInMemoryRetriever,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    query = "Do you like hot weather?"
    documents = (
        hybrid_asymmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    )
    assert in_memory_retriever_documents[0] == to_document(documents[0].document_chunk)
    assert len(documents) <= 2


def test_symmetric_in_memory_retriever(
    hybrid_symmetric_in_memory_retriever: HybridQdrantInMemoryRetriever,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    query = "I hate drizzle"
    documents = hybrid_symmetric_in_memory_retriever.get_relevant_documents_with_scores(
        query
    )
    assert in_memory_retriever_documents[1] == to_document(documents[0].document_chunk)
    assert len(documents) <= 2


def test_hybrid_in_memory_retriever(
    hybrid_asymmetric_in_memory_retriever: HybridQdrantInMemoryRetriever,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    query = "Summer rain"
    documents = (
        hybrid_asymmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    )
    assert in_memory_retriever_documents[3] == to_document(documents[0].document_chunk)
    assert len(documents) <= 2
