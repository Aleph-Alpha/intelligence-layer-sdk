from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors.retrievers.in_memory_retriever import (
    InMemoryRetriever,
)


@fixture
def in_memory_retriever_texts() -> Sequence[str]:
    return ["I do not like rain", "Summer is warm", "We are so back"]


def test_asymmetric_in_memory(
    asymmetric_in_memory_retriever: InMemoryRetriever,
    in_memory_retriever_texts: Sequence[str],
) -> None:
    query = "Do you like summer?"
    documents = asymmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    assert in_memory_retriever_texts[1] == documents[0].text
    assert len(documents) <= 2


def test_symmetric_in_memory(
    symmetric_in_memory_retriever: InMemoryRetriever,
    in_memory_retriever_texts: Sequence[str],
) -> None:
    query = "I hate drizzle"
    documents = symmetric_in_memory_retriever.get_relevant_documents_with_scores(query)
    assert in_memory_retriever_texts[0] == documents[0].text
    assert len(documents) <= 2
