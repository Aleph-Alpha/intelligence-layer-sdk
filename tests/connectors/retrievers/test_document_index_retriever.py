import pytest

from intelligence_layer.connectors.retrievers.document_index_retriever import (
    DocumentIndexRetriever,
)


@pytest.mark.internal
def test_document_index_retriever(
    document_index_retriever: DocumentIndexRetriever,
) -> None:
    documents = document_index_retriever.get_relevant_documents_with_scores("Coca-Cola")
    assert len(documents) > 0
