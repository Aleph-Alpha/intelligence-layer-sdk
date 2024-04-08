from typing import Generic, Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import (
    ID,
    BaseRetriever,
    SearchResult,
)
from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.evaluation import (
    EvaluationLogic,
    Example,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic


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


class ExpectedSearchOutput(BaseModel):
    document_id: str
    start_idx: int
    end_idx: int
    origin_chunk: str
    answer: str
    task_label: str


class SearchEvaluation(BaseModel):
    rank: Optional[int]
    similarity_score: Optional[float]


class SearchEvaluationLogic(
    EvaluationLogic[
        SearchInput, SearchOutput[ID], ExpectedSearchOutput, SearchEvaluation
    ]
):
    def do_evaluate(
        self,
        example: Example[SearchInput, ExpectedSearchOutput],
        *output: SuccessfulExampleOutput[SearchOutput[ID]],
    ) -> SearchEvaluation:
        assert len(output) == 1
        results = output[0].output.results

        def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
            a_start, a_end = a
            b_start, b_end = b
            return a_start < b_end and b_start < a_end

        index, score = next(
            (
                (index, result.score)
                for index, result in enumerate(results)
                if overlaps(
                    (result.document_chunk.start, result.document_chunk.end),
                    (
                        example.expected_output.start_idx,
                        example.expected_output.end_idx,
                    ),
                )
            ),
            (None, None),
        )

        return SearchEvaluation(rank=index, similarity_score=score)


class MeanTopK(BaseModel):
    top_k: int
    mean: float


class AggregatedSearchEvaluation(BaseModel):
    mean_rank: float
    mean_reciprocal_rank: float
    mean_top_ks: Sequence[MeanTopK]


class SearchAggregationLogic(
    AggregationLogic[SearchEvaluation, AggregatedSearchEvaluation]
):
    pass
