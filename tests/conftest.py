from os import getenv
from pathlib import Path
from typing import Sequence, cast

from aleph_alpha_client import Image
from dotenv import load_dotenv
from faker import Faker
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors.document_index.document_index import (
    DocumentIndexClient,
)
from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
    LimitedConcurrencyClient,
)
from intelligence_layer.connectors.retrievers.base_retriever import (
    Document,
    DocumentChunk,
)
from intelligence_layer.connectors.retrievers.document_index_retriever import (
    DocumentIndexRetriever,
)
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
    RetrieverType,
)
from intelligence_layer.core import NoOpTracer, Task, TaskSpan
from intelligence_layer.evaluation import (
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
)
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    InMemoryAggregationRepository,
)


@fixture(scope="session")
def token() -> str:
    load_dotenv()
    token = getenv("AA_TOKEN")
    assert isinstance(token, str)
    return token


@fixture(scope="session")
def client(token: str) -> AlephAlphaClientProtocol:
    """Provide fixture for api."""
    return LimitedConcurrencyClient.from_token(token=token)


@fixture
def no_op_tracer() -> NoOpTracer:
    return NoOpTracer()


@fixture(scope="session")
def prompt_image() -> Image:
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    return cast(Image, Image.from_file(image_source_path))  # from_file lacks type-hint


@fixture
def asymmetric_in_memory_retriever(
    client: AlephAlphaClientProtocol,
    in_memory_retriever_documents: Sequence[Document],
) -> QdrantInMemoryRetriever:
    return QdrantInMemoryRetriever(
        client,
        in_memory_retriever_documents,
        k=2,
        retriever_type=RetrieverType.ASYMMETRIC,
    )


@fixture
def symmetric_in_memory_retriever(
    client: AlephAlphaClientProtocol,
    in_memory_retriever_documents: Sequence[Document],
) -> QdrantInMemoryRetriever:
    return QdrantInMemoryRetriever(
        client,
        in_memory_retriever_documents,
        k=2,
        retriever_type=RetrieverType.SYMMETRIC,
    )


@fixture
def document_index(token: str) -> DocumentIndexClient:
    return DocumentIndexClient(token)


@fixture
def document_index_retriever(
    document_index: DocumentIndexClient,
) -> DocumentIndexRetriever:
    return DocumentIndexRetriever(
        document_index, namespace="aleph-alpha", collection="wikipedia-de", k=2
    )


def to_document(document_chunk: DocumentChunk) -> Document:
    return Document(text=document_chunk.text, metadata=document_chunk.metadata)


class DummyStringInput(BaseModel):
    input: str

    @classmethod
    def any(cls) -> "DummyStringInput":
        fake = Faker()
        return cls(input=fake.text())


class DummyStringOutput(BaseModel):
    output: str

    @classmethod
    def any(cls) -> "DummyStringOutput":
        fake = Faker()
        return cls(output=fake.text())


class DummyStringTask(Task[DummyStringInput, DummyStringOutput]):
    def do_run(self, input: DummyStringInput, task_span: TaskSpan) -> DummyStringOutput:
        return DummyStringOutput.any()


@fixture
def dummy_string_task() -> DummyStringTask:
    return DummyStringTask()


@fixture
def in_memory_dataset_repository() -> InMemoryDatasetRepository:
    return InMemoryDatasetRepository()


@fixture
def in_memory_run_repository() -> InMemoryRunRepository:
    return InMemoryRunRepository()


@fixture
def in_memory_evaluation_repository() -> InMemoryEvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def in_memory_aggregation_repository() -> InMemoryAggregationRepository:
    return InMemoryAggregationRepository()
