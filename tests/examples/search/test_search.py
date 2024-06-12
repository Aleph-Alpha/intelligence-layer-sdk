from collections.abc import Sequence
from statistics import mean

from pytest import fixture

from intelligence_layer.connectors import (
    Document,
    DocumentChunk,
    QdrantInMemoryRetriever,
    SearchResult,
)
from intelligence_layer.core import NoOpTracer
from intelligence_layer.evaluation import Example
from intelligence_layer.examples import (
    ExpectedSearchOutput,
    Search,
    SearchAggregationLogic,
    SearchEvaluation,
    SearchEvaluationLogic,
    SearchInput,
    SearchOutput,
)
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


@fixture
def expected_output() -> ExpectedSearchOutput[str]:
    return ExpectedSearchOutput(
        document_id="1",
        start_idx=0,
        end_idx=5,
    )


@fixture
def example(
    expected_output: ExpectedSearchOutput[str],
) -> Example[SearchInput, ExpectedSearchOutput[str]]:
    return Example(input=SearchInput(query=""), expected_output=expected_output)


@fixture
def search_eval_logic() -> SearchEvaluationLogic[str]:
    return SearchEvaluationLogic[str]()


@fixture
def search_evaluations() -> Sequence[SearchEvaluation]:
    return [
        SearchEvaluation(rank=1, similarity_score=0.7),
        SearchEvaluation(rank=3, similarity_score=0.6),
        SearchEvaluation(rank=10, similarity_score=0.5),
        SearchEvaluation(rank=None, similarity_score=None),
        SearchEvaluation(rank=None, similarity_score=None),
    ]


@fixture
def search_aggregation_logic() -> SearchAggregationLogic:
    return SearchAggregationLogic(top_ks_to_evaluate=[1, 3])


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
    assert (
        result.results[0].document_chunk.end
        == len(in_memory_retriever_documents[2].text) - 1
    )


def test_search_evaluation_logic_works_for_overlapping_output(
    example: Example[SearchInput, ExpectedSearchOutput[str]],
    search_eval_logic: SearchEvaluationLogic[str],
) -> None:
    output = SearchOutput(
        results=[
            SearchResult(
                id="1",
                score=0.5,
                document_chunk=DocumentChunk(text="llo", start=2, end=5),
            )
        ]
    )
    eval = search_eval_logic.do_evaluate_single_output(example, output)

    assert eval.rank == 1
    assert eval.similarity_score == output.results[0].score


def test_search_evaluation_logic_works_for_wholly_included_output(
    example: Example[SearchInput, ExpectedSearchOutput[str]],
    search_eval_logic: SearchEvaluationLogic[str],
) -> None:
    output = SearchOutput(
        results=[
            SearchResult(
                id="1",
                score=0.5,
                document_chunk=DocumentChunk(text="l", start=2, end=3),
            )
        ]
    )
    eval = search_eval_logic.do_evaluate_single_output(example, output)

    assert eval.rank == 1
    assert eval.similarity_score == output.results[0].score


def test_search_evaluation_logic_works_for_identical_ranges(
    example: Example[SearchInput, ExpectedSearchOutput[str]],
    search_eval_logic: SearchEvaluationLogic[str],
) -> None:
    output = SearchOutput(
        results=[
            SearchResult(
                id="1",
                score=0.5,
                document_chunk=DocumentChunk(text="hallo", start=0, end=5),
            )
        ]
    )
    eval = search_eval_logic.do_evaluate_single_output(example, output)

    assert eval.rank == 1
    assert eval.similarity_score == output.results[0].score


def test_search_evaluation_logic_works_for_non_overlapping_output(
    example: Example[SearchInput, ExpectedSearchOutput[str]],
    search_eval_logic: SearchEvaluationLogic[str],
) -> None:
    output = SearchOutput(
        results=[
            SearchResult(
                id="1",
                score=0.5,
                document_chunk=DocumentChunk(text=" test.", start=5, end=10),
            )
        ]
    )
    eval = search_eval_logic.do_evaluate_single_output(example, output)

    assert not eval.rank
    assert not eval.similarity_score


def test_search_aggregation_logic_works(
    search_evaluations: Sequence[SearchEvaluation],
    search_aggregation_logic: SearchAggregationLogic,
) -> None:
    aggregations = search_aggregation_logic.aggregate(search_evaluations)

    assert (
        aggregations.mean_score
        == mean(
            [
                eval.similarity_score
                for eval in search_evaluations
                if eval.similarity_score
            ]
        )
        == 0.6
    )
    assert (
        round(aggregations.mean_reciprocal_rank, 5)
        == round(mean([1 / eval.rank for eval in search_evaluations if eval.rank]), 5)
        == round((1 + (1 / 3) + (1 / 10)) / 3, 5)
    )
    assert aggregations.mean_top_ks
    assert aggregations.chunk_found.found_count == 3
    assert aggregations.chunk_found.expected_count == len(search_evaluations) == 5
    assert aggregations.chunk_found.percentage == 3 / 5
    assert aggregations.mean_top_ks[1] == 1 / 3
    assert aggregations.mean_top_ks[3] == 2 / 3
