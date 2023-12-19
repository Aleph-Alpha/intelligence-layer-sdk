from typing import Generic, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import (
    ID,
    BaseRetriever,
    SearchResult,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan


class SearchInput(BaseModel):
    """The input for a `Search` task.

    Attributes:
        query: The text to be searched with.
    """

    query: str


class SearchOutput(BaseModel, Generic[ID]):
    """The output of a `Search` task.

    Attributes:
        results: Each result contains a text and corresponding score.
    """

    results: Sequence[SearchResult[ID]]


class Search(Generic[ID], Task[SearchInput, SearchOutput[ID]]):
    """Performs search to find documents.

    Given a query, this task will utilize a retriever to fetch relevant text search results.
    Each result consists of a string representation of the content and an associated score
    indicating its relevance to the provided query.

    Args:
        retriever: Implements logic to retrieve matching texts to the query.

    Example:
        >>> from os import getenv
        >>> from intelligence_layer.connectors import (
        ...     DocumentIndexClient,
        ... )
        >>> from intelligence_layer.connectors import (
        ...     DocumentIndexRetriever,
        ... )
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.use_cases import Search, SearchInput


        >>> document_index = DocumentIndexClient(getenv("AA_TOKEN"))
        >>> retriever = DocumentIndexRetriever(document_index, "aleph-alpha", "wikipedia-de", 3)
        >>> task = Search(retriever)
        >>> input = SearchInput(query="When did East and West Germany reunite?")
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    def __init__(self, retriever: BaseRetriever[ID]):
        super().__init__()
        self._retriever = retriever

    def do_run(self, input: SearchInput, task_span: TaskSpan) -> SearchOutput[ID]:
        results = self._retriever.get_relevant_documents_with_scores(input.query)
        return SearchOutput(results=results)
