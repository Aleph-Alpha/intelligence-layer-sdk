from typing import Sequence
from pytest import fixture
from intelligence_layer.retrievers.in_memory import InMemoryRetriever


QUERY = "Do you like summer?"

@fixture
def in_memory_retriever_texts() -> Sequence[str]:
    return ["I do not like rain", "Summer is warm", "We are so back"]


def test_in_memory(in_memory_retriever: InMemoryRetriever, in_memory_retriever_texts: Sequence[str]) -> None:
    documents = in_memory_retriever.get_relevant_documents_with_scores(
        QUERY
    )
    assert in_memory_retriever_texts[1] == documents[0].text
    assert len(documents) <= 2
