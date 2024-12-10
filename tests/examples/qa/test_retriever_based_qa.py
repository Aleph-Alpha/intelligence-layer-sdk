import pytest
from pytest import fixture

from intelligence_layer.connectors.document_index.document_index import DocumentPath
from intelligence_layer.connectors.retrievers.document_index_retriever import (
    DocumentIndexRetriever,
)
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core import NoOpTracer
from intelligence_layer.examples import (
    MultipleChunkQa,
    RetrieverBasedQa,
    RetrieverBasedQaInput,
)


@fixture
def retriever_based_qa_with_in_memory_retriever(
    multiple_chunk_qa: MultipleChunkQa,
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
) -> RetrieverBasedQa[int]:
    return RetrieverBasedQa(
        retriever=asymmetric_in_memory_retriever, multi_chunk_qa=multiple_chunk_qa
    )


@fixture
def retriever_based_qa_with_document_index(
    multiple_chunk_qa: MultipleChunkQa, document_index_retriever: DocumentIndexRetriever
) -> RetrieverBasedQa[DocumentPath]:
    return RetrieverBasedQa(
        retriever=document_index_retriever, multi_chunk_qa=multiple_chunk_qa
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_retriever_based_qa_using_in_memory_retriever(
    retriever_based_qa_with_in_memory_retriever: RetrieverBasedQa[int],
    no_op_tracer: NoOpTracer,
) -> None:
    question = "When was Robert Moses born?"
    input = RetrieverBasedQaInput(question=question)
    output = retriever_based_qa_with_in_memory_retriever.run(input, no_op_tracer)
    assert output.answer
    assert "1888" in output.answer
    assert output.subanswers[0].id == 3
