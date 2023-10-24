from pytest import fixture
from intelligence_layer.retrievers.in_memory import InMemoryRetriever
from aleph_alpha_client import Client
from intelligence_layer.task import Chunk, NoOpDebugLogger

CHUNKS = [Chunk(t) for t in ["I do not like rain", "Summer is warm", "We are so back"]]
QUERY = "Do you like summer?"


@fixture
def retriever(client: Client) -> InMemoryRetriever:
    return InMemoryRetriever(client, CHUNKS)


def test_in_memory(retriever: InMemoryRetriever) -> None:
    documents = retriever.get_relevant_documents_with_scores(
        QUERY, NoOpDebugLogger(), k=2
    )
    assert CHUNKS[1] == documents[0].text
    assert len(documents) <= 2
