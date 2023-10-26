from pytest import fixture
from typing import Sequence

from intelligence_layer.connectors.retrievers.in_memory_retriever import (
    InMemoryRetriever,
)
from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.use_cases.search.search import Search, SearchInput


@fixture
def in_memory_retriever_texts() -> Sequence[str]:
    return ["I do not like rain", "Summer is warm", "We are so back"]


@fixture
def search(in_memory_retriever: InMemoryRetriever) -> Search:
    return Search(in_memory_retriever)


def test_search(
    search: Search,
    no_op_debug_logger: NoOpDebugLogger,
    in_memory_retriever_texts: Sequence[str],
) -> None:
    search_input = SearchInput(query="Are we so back?")
    result = search.run(search_input, no_op_debug_logger)
    assert [r.text for r in result.results] == [in_memory_retriever_texts[2]]
