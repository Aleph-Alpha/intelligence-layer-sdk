from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.search.search import Search, SearchInput


@fixture
def in_memory_retriever_documents() -> Sequence[Document]:
    return [
        Document(text="I do not like rain", id=""),
        Document(text="Summer is warm", id=""),
        Document(text="We are so back", id=""),
    ]


@fixture
def search(asymmetric_in_memory_retriever: QdrantInMemoryRetriever) -> Search:
    return Search(asymmetric_in_memory_retriever)


def test_search(
    search: Search,
    no_op_tracer: NoOpTracer,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    search_input = SearchInput(query="Are we so back?")
    result = search.run(search_input, no_op_tracer)
    assert [r.document for r in result.results] == [in_memory_retriever_documents[2]]
