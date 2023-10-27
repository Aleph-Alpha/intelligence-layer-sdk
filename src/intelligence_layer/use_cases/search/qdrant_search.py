from pydantic import BaseModel
from qdrant_client.http.models.models import Filter

from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.use_cases.search.search import SearchOutput


class QdrantSearchInput(BaseModel):
    """The input for a `FilterSearch` task.

    Attributes:
        query: The text to be searched with.
        filter: Conditions to filter by as offered by Qdrant.
    """

    query: str
    filter: Filter


class QdrantSearch(Task[QdrantSearchInput, SearchOutput]):
    """Performs search to find documents using QDrant filtering methods.

    Given a query, this task will utilize a retriever to fetch relevant text search results.
    Contrary to `Search`, this `Task` offers the option to filter.

    Args:
        in_memory_retriever: Implements logic to retrieve matching texts to the query.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> documents = [
        >>>  Document(
        >>>         text="West and East Germany reunited in 1990.
        >>>         metadata={"title": "Germany"}
        >>>     )
        >>> ]
        >>> retriever = InMemoryRetriever(client, documents)
        >>> task = FilterSearch(retriever)
        >>> input = FilterSearchInput(
        >>>     query="When did East and West Germany reunite?"
        >>>     filter=models.Filter(
        >>>         must=[
        >>>             models.FieldCondition(
        >>>                 key="metadata.title",
        >>>                 match="Germany",
        >>>             ),
        >>>         ]
        >>>     )
        >>> )
        >>> logger = InMemoryLogger(name="Filter Search")
        >>> output = task.run(input, logger)
    """

    def __init__(self, in_memory_retriever: QdrantInMemoryRetriever):
        super().__init__()
        self._in_memory_retriever = in_memory_retriever

    def run(self, input: QdrantSearchInput, logger: DebugLogger) -> SearchOutput:
        results = self._in_memory_retriever.get_filtered_documents_with_scores(
            input.query, input.filter
        )
        return SearchOutput(results=results)
