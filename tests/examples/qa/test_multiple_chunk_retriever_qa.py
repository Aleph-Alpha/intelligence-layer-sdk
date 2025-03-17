from pytest import fixture

from intelligence_layer.connectors import QdrantInMemoryRetriever
from intelligence_layer.core import LuminousControlModel, NoOpTracer
from intelligence_layer.core.tracer.in_memory_tracer import InMemoryTracer
from intelligence_layer.examples import (
    ExpandChunks,
    MultipleChunkRetrieverQa,
    RetrieverBasedQaInput,
)


@fixture
def multiple_chunk_retriever_qa(
    llama_control_model: LuminousControlModel,
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
) -> MultipleChunkRetrieverQa[int]:
    return MultipleChunkRetrieverQa(
        retriever=asymmetric_in_memory_retriever,
        model=llama_control_model,
        expand_chunks=ExpandChunks(
            asymmetric_in_memory_retriever, llama_control_model, 256
        ),
    )


def test_multiple_chunk_retriever_qa_using_in_memory_retriever(
    multiple_chunk_retriever_qa: MultipleChunkRetrieverQa[int],
    no_op_tracer: NoOpTracer,
) -> None:
    question = "When was Robert Moses born?"
    input = RetrieverBasedQaInput(question=question)
    tracer = InMemoryTracer()
    output = multiple_chunk_retriever_qa.run(input, tracer)
    assert output.answer
    assert "1888" in output.answer
    assert len(output.sources) == 5
