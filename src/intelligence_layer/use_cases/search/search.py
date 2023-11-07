from typing import Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import (
    BaseRetriever,
    SearchResult,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import Tracer


class SearchInput(BaseModel):
    """The input for a `Search` task.

    Attributes:
        query: The text to be searched with.
    """

    query: str


class SearchOutput(BaseModel):
    """The output of a `Search` task.

    Attributes:
        results: Each result contains a text and corresponding score.
    """

    results: Sequence[SearchResult]


class Search(Task[SearchInput, SearchOutput]):
    """Performs search to find documents.

    Given a query, this task will utilize a retriever to fetch relevant text search results.
    Each result consists of a string representation of the content and an associated score
    indicating its relevance to the provided query.

    Args:
        retriever: Implements logic to retrieve matching texts to the query.

    Example:
        >>> document_index = DocumentIndex(token)
        >>> retriever = DocumentIndexRetriever(document_index, "my_namespace", \
            "country_facts_collection", 3)
        >>> task = Search(retriever)
        >>> input = SearchInput(
                query="When did East and West Germany reunite?"
            )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
        >>> print(output.results[0].text[-5:])
        1990.
    """

    def __init__(self, retriever: BaseRetriever):
        super().__init__()
        self._retriever = retriever

    def run(self, input: SearchInput, tracer: Tracer) -> SearchOutput:
        results = self._retriever.get_relevant_documents_with_scores(input.query)
        return SearchOutput(results=results)
