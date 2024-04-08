from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors import Document, SearchResult, QdrantInMemoryRetriever, DocumentChunk
from intelligence_layer.core import NoOpTracer
from intelligence_layer.evaluation import Example
from intelligence_layer.use_cases import Search, SearchEvaluationLogic, SearchInput, ExpectedSearchOutput, SearchOutput
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
def example() -> Example:
    return Example(input=SearchInput(query=""))


@fixture
def expected_output() -> ExpectedSearchOutput:
    return ExpectedSearchOutput(
        document_id="1",
        start_idx=0,
        end_idx=5,
        origin_chunk="hallo ",
        answer="",
        task_label=""
    )


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


def test_search_evaluation_logic_works_for_non_overlapping_output(example: Example, expected_output: ExpectedSearchOutput) -> None:
    logic = SearchEvaluationLogic()
    output = SearchOutput(
        results=[
            SearchResult(
                id="1",
                score=0.5,
                document_chunk=DocumentChunk(
                    text="test ",
                    start=5,
                    end=10
                )
            )
        ]
    )
