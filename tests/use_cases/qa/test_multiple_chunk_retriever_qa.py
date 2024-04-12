from pytest import fixture

from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core import NoOpTracer
from intelligence_layer.use_cases import MultipleChunkRetrieverQa, RetrieverBasedQaInput


@fixture
def multiple_chunk_retriever_qa(
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
) -> MultipleChunkRetrieverQa[int]:
    return MultipleChunkRetrieverQa(retriever=asymmetric_in_memory_retriever)


def test_retriever_based_qa_using_in_memory_retriever(
    multiple_chunk_retriever_qa: MultipleChunkRetrieverQa[int],
    no_op_tracer: NoOpTracer,
) -> None:
    question = "When was Robert Moses born?"
    input = RetrieverBasedQaInput(question=question)
    output = multiple_chunk_retriever_qa.run(input, no_op_tracer)
    assert output.answer
    assert "1888" in output.answer
    assert len(output.sources) == 2
