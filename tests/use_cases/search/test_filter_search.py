from pytest import fixture
from typing import Sequence

from qdrant_client.http.models import models

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.in_memory_retriever import (
    InMemoryRetriever,
)
from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.use_cases.search.filter_search import (
    FilterSearch,
    FilterSearchInput,
)


@fixture
def in_memory_retriever_documents() -> Sequence[Document]:
    return [
        Document(
            text="Germany reunited. I kind of fit and am of the correct type.",
            metadata={"type": "doc"},
        ),
        Document(
            text="Cats are small animals. Well, I do not fit at all but I am of the correct type.",
            metadata={"type": "doc"},
        ),
        Document(
            text="Germany reunited in 1990. This document fits perfectly but it is of the wrong type.",
            metadata={"type": "no doc"},
        ),
    ]


@fixture
def filter_search(asymmetric_in_memory_retriever: InMemoryRetriever) -> FilterSearch:
    return FilterSearch(asymmetric_in_memory_retriever)


def test_filter_search(
    filter_search: FilterSearch,
    no_op_debug_logger: NoOpDebugLogger,
    in_memory_retriever_documents: Sequence[Document],
) -> None:
    search_input = FilterSearchInput(
        query="When did Germany reunite?",
        limit=1,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.type",
                    match=models.MatchValue(value="doc"),
                ),
            ]
        ),
    )
    result = filter_search.run(search_input, no_op_debug_logger)
    assert [r.document for r in result.results] == [in_memory_retriever_documents[0]]
