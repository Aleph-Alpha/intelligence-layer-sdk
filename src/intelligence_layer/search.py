from typing import Sequence
from pydantic import BaseModel
from intelligence_layer.retrievers.base import BaseRetriever, SearchResult
from intelligence_layer.task import DebugLogger, Task


class SearchInput(BaseModel):
    query: str


class SearchOutput(BaseModel):
    results: Sequence[SearchResult]


class Search(Task[SearchInput, SearchOutput]):
    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever

    def run(self, input: SearchInput, debug: DebugLogger) -> SearchOutput:
        results = self._retriever.get_relevant_documents_with_scores(input.query)
        return SearchOutput(results=results)
