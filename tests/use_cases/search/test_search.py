from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors import (
    Document,
    DocumentChunk,
    QdrantInMemoryRetriever,
    SearchResult,
)
from intelligence_layer.core import NoOpTracer
from intelligence_layer.evaluation import Example
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput
from intelligence_layer.use_cases import (
    ExpectedSearchOutput,
    Search,
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
def expected_output() -> ExpectedSearchOutput:
    return ExpectedSearchOutput(
        document_id="1",
        start_idx=0,
        end_idx=5,
        origin_chunk="hallo",
        answer="",
        task_label="",
    )


@fixture
def example(
    expected_output: ExpectedSearchOutput,
) -> Example[SearchInput, ExpectedSearchOutput]:
    return Example(input=SearchInput(query=""), expected_output=expected_output)


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
    example: Example[SearchInput, ExpectedSearchOutput],
) -> None:
    logic = SearchEvaluationLogic[SearchResult[str]]()
    output = SuccessfulExampleOutput(
        run_id="1",
        example_id="1",
        output=SearchOutput(
            results=[
                SearchResult[str](
                    id="1",
                    score=0.5,
                    document_chunk=DocumentChunk(text="llo", start=2, end=5),
                )
            ]
        ),
    )
    eval = logic.do_evaluate(example, output)

    assert eval.rank == 0
    assert eval.similarity_score == output.output.results[0].score


def test_search_evaluation_logic_works_for_wholly_included_output(
    example: Example[SearchInput, ExpectedSearchOutput],
) -> None:
    logic = SearchEvaluationLogic()
    output = SuccessfulExampleOutput(
        run_id="1",
        example_id="1",
        output=SearchOutput(
            results=[
                SearchResult(
                    id="1",
                    score=0.5,
                    document_chunk=DocumentChunk(text="l", start=2, end=3),
                )
            ]
        ),
    )
    eval = logic.do_evaluate(example, *[output])

    assert eval.rank == 0
    assert eval.similarity_score == output.output.results[0].score


def test_search_evaluation_logic_works_for_identical_ranges(
    example: Example[SearchInput, ExpectedSearchOutput],
) -> None:
    logic = SearchEvaluationLogic()
    output = SuccessfulExampleOutput(
        run_id="1",
        example_id="1",
        output=SearchOutput(
            results=[
                SearchResult(
                    id="1",
                    score=0.5,
                    document_chunk=DocumentChunk(text="hallo", start=0, end=5),
                )
            ]
        ),
    )
    eval = logic.do_evaluate(example, *[output])

    assert eval.rank == 0
    assert eval.similarity_score == output.output.results[0].score


def test_search_evaluation_logic_works_for_non_overlapping_output(
    example: Example[SearchInput, ExpectedSearchOutput],
) -> None:
    logic = SearchEvaluationLogic()
    output = SuccessfulExampleOutput(
        run_id="1",
        example_id="1",
        output=SearchOutput(
            results=[
                SearchResult(
                    id="1",
                    score=0.5,
                    document_chunk=DocumentChunk(text=" test.", start=5, end=10),
                )
            ]
        ),
    )
    eval = logic.do_evaluate(example, *[output])

    assert not eval.rank
    assert not eval.similarity_score
