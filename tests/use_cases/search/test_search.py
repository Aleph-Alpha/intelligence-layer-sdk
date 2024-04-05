from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core import NoOpTracer
from intelligence_layer.use_cases.search.search import Search, SearchInput
from tests.conftest import to_document


@fixture
def in_memory_retriever_documents() -> Sequence[Document]:
    return [
        Document(text="I do not like rain"),
        Document(text="Summer is warm"),
        Document(text="We are so back"),
    ]


@fixture
def search(asymmetric_in_memory_retriever: QdrantInMemoryRetriever) -> Search[int]:
    return Search(asymmetric_in_memory_retriever)


def test_search(
    search: Search[int],
    no_op_tracer: NoOpTracer,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    search_input = SearchInput(query="Are we so back?")
    result = search.run(search_input, no_op_tracer)
    assert [to_document(r.document_chunk) for r in result.results] == [
        in_memory_retriever_documents[2]
    ]
    assert result.results[0].document_chunk.start == 0
    assert result.results[0].document_chunk.end == len(in_memory_retriever_documents[2].text) - 1
