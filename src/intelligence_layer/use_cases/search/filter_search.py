from pydantic import BaseModel
from qdrant_client.http.models.models import Filter

from intelligence_layer.connectors.retrievers.in_memory_retriever import InMemoryRetriever
from intelligence_layer.core.task import Task
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.use_cases.search.search import SearchOutput


class FilterSearchInput(BaseModel):
    """The input for a `Search` task.

    Attributes:
        query: The text to be searched with.
        limit: XXX
        filter: XXX
    """

    query: str
    limit: int
    filter: Filter


class FilterSearch(Task[FilterSearchInput, SearchOutput]):
    """Performs search to find documents using QDrant filtering methods.

    Given a query, this task will utilize a retriever to fetch relevant text search results.
    Each result consists of a string representation of the content and an associated score
    indicating its relevance to the provided query.
XXX
    Args:
        in_memory_retriever: Implements logic to retrieve matching texts to the query.

    Example:
        >>> document_index = DocumentIndex(token)
        >>> retriever = DocumentIndexRetriever(document_index, "my_namespace", "country_facts_collection", 3)
        >>> task = Search(retriever)
        >>> input = SearchInput(
        >>>     query="When did East and West Germany reunite?"
        >>> )
        >>> logger = InMemoryLogger(name="Search")
        >>> output = task.run(input, logger)
        >>> print(output.results[0].text[-5:])
        1990.
    """

    def __init__(self, in_memory_retriever: InMemoryRetriever):
        super().__init__()
        self._in_memory_retriever = in_memory_retriever

    def run(self, input: FilterSearchInput, logger: DebugLogger) -> SearchOutput:
        results = self._in_memory_retriever.get_filtered_documents_with_scores(input.query, input.limit, input.filter)
        return SearchOutput(results=results)
