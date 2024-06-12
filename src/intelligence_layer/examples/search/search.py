from collections.abc import Iterable, Mapping, Sequence
from typing import Generic, Optional

from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import (
    ID,
    BaseRetriever,
    SearchResult,
)
from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.evaluation import (
    AggregationLogic,
    Example,
    MeanAccumulator,
    SingleOutputEvaluationLogic,
)


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
        >>> from intelligence_layer.examples import Search, SearchInput


        >>> document_index = DocumentIndexClient(getenv("AA_TOKEN"))
        >>> retriever = DocumentIndexRetriever(document_index, "asymmetric", "aleph-alpha", "wikipedia-de", 3)
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


class ExpectedSearchOutput(BaseModel, Generic[ID]):
    document_id: ID
    start_idx: int
    end_idx: int


class SearchEvaluation(BaseModel):
    rank: Optional[int]
    similarity_score: Optional[float]


class SearchEvaluationLogic(
    Generic[ID],
    SingleOutputEvaluationLogic[
        SearchInput, SearchOutput[ID], ExpectedSearchOutput[ID], SearchEvaluation
    ],
):
    def do_evaluate_single_output(
        self,
        example: Example[SearchInput, ExpectedSearchOutput[ID]],
        output: SearchOutput[ID],
    ) -> SearchEvaluation:
        results = output.results

        def same_document(id_a: ID, id_b: ID) -> bool:
            return id_a == id_b

        def chunks_overlap(range_a: tuple[int, int], range_b: tuple[int, int]) -> bool:
            a_start, a_end = range_a
            b_start, b_end = range_b
            return a_start < b_end and b_start < a_end

        rank, score = next(
            (
                (index + 1, result.score)
                for index, result in enumerate(results)
                if same_document(result.id, example.expected_output.document_id)
                and chunks_overlap(
                    (result.document_chunk.start, result.document_chunk.end),
                    (
                        example.expected_output.start_idx,
                        example.expected_output.end_idx,
                    ),
                )
            ),
            (None, None),
        )

        return SearchEvaluation(rank=rank, similarity_score=score)


class ChunkFound(BaseModel):
    found_count: int  # found => chunk was within top-k results of retriever
    expected_count: int
    percentage: float


class AggregatedSearchEvaluation(BaseModel):
    mean_score: float
    mean_reciprocal_rank: float
    mean_top_ks: Mapping[int, float]
    chunk_found: ChunkFound


class SearchAggregationLogic(
    AggregationLogic[SearchEvaluation, AggregatedSearchEvaluation]
):
    def __init__(self, top_ks_to_evaluate: Sequence[int]) -> None:
        assert all(top_k > 0 for top_k in top_ks_to_evaluate)
        self.top_ks_to_evaluate = top_ks_to_evaluate

    def aggregate(
        self, evaluations: Iterable[SearchEvaluation]
    ) -> AggregatedSearchEvaluation:
        score_accumulator = MeanAccumulator()
        reciprocal_rank_accumulator = MeanAccumulator()
        chunk_found_accumulator = MeanAccumulator()
        top_k_accumulator = {
            top_k: MeanAccumulator() for top_k in self.top_ks_to_evaluate
        }

        for evaluation in evaluations:
            chunk_found = bool(evaluation.rank)
            chunk_found_accumulator.add(chunk_found)
            if chunk_found:
                assert evaluation.similarity_score and evaluation.rank

                score_accumulator.add(evaluation.similarity_score)
                reciprocal_rank_accumulator.add(1 / evaluation.rank)
                for top_k in self.top_ks_to_evaluate:
                    top_k_accumulator[top_k].add(
                        1.0 if evaluation.rank <= top_k else 0.0
                    )

        return AggregatedSearchEvaluation(
            mean_score=score_accumulator.extract(),
            mean_reciprocal_rank=reciprocal_rank_accumulator.extract(),
            mean_top_ks={
                top_k: acc.extract() for top_k, acc in top_k_accumulator.items()
            },
            chunk_found=ChunkFound(
                found_count=int(chunk_found_accumulator._acc),
                expected_count=chunk_found_accumulator._n,
                percentage=chunk_found_accumulator.extract(),
            ),
        )
