import os
from collections.abc import Sequence
from os import getenv
from pathlib import Path
from typing import cast

from aleph_alpha_client import Client, Image
from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.connectors import (
    AlephAlphaClientProtocol,
    Document,
    LimitedConcurrencyClient,
    QdrantInMemoryRetriever,
    RetrieverType,
)
from intelligence_layer.core import (
    Llama3InstructModel,
    LuminousControlModel,
    NoOpTracer,
    Pharia1ChatModel,
    utc_now,
)
from intelligence_layer.evaluation import (
    AsyncInMemoryEvaluationRepository,
    EvaluationOverview,
    InMemoryAggregationRepository,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    RunOverview,
)
from tests.conftest_document_index import *  # noqa: F403 - we import everything here to get the file to be "appended" to this file and thus making all fixtures available


@fixture(scope="session")
def token() -> str:
    load_dotenv()
    token = getenv("AA_TOKEN")
    assert isinstance(token, str)
    return token


@fixture(scope="session")
def inference_url() -> str:
    return os.environ["CLIENT_URL"]


@fixture(scope="session")
def client(token: str, inference_url: str) -> AlephAlphaClientProtocol:
    return LimitedConcurrencyClient(
        Client(token, host=inference_url),
        max_concurrency=10,
        max_retry_time=10,
    )


@fixture(scope="session")
def llama_control_model(client: AlephAlphaClientProtocol) -> Llama3InstructModel:
    return Llama3InstructModel("llama-3.1-8b-instruct", client)


@fixture(scope="session")
def luminous_control_model(client: AlephAlphaClientProtocol) -> LuminousControlModel:
    return LuminousControlModel("luminous-base-control", client)


@fixture(scope="session")
def pharia_1_chat_model(client: AlephAlphaClientProtocol) -> Pharia1ChatModel:
    return Pharia1ChatModel("pharia-1-llm-7b-control", client)


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
        in_memory_retriever_documents,
        client=client,
        k=2,
        retriever_type=RetrieverType.ASYMMETRIC,
    )


@fixture
def symmetric_in_memory_retriever(
    client: AlephAlphaClientProtocol,
    in_memory_retriever_documents: Sequence[Document],
) -> QdrantInMemoryRetriever:
    return QdrantInMemoryRetriever(
        in_memory_retriever_documents,
        client=client,
        k=2,
        retriever_type=RetrieverType.SYMMETRIC,
    )


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


@fixture()
def async_in_memory_evaluation_repository() -> AsyncInMemoryEvaluationRepository:
    return AsyncInMemoryEvaluationRepository()


@fixture
def run_overview() -> RunOverview:
    return RunOverview(
        dataset_id="dataset-id",
        id="run-id-1",
        start=utc_now(),
        end=utc_now(),
        failed_example_count=0,
        successful_example_count=3,
        description="test run overview 1",
        labels=set(),
        metadata=dict(),
    )


@fixture
def evaluation_id() -> str:
    return "evaluation-id-1"


@fixture
def evaluation_overview(
    evaluation_id: str, run_overview: RunOverview
) -> EvaluationOverview:
    return EvaluationOverview(
        id=evaluation_id,
        start_date=utc_now(),
        end_date=utc_now(),
        successful_evaluation_count=1,
        failed_evaluation_count=1,
        run_overviews=frozenset([run_overview]),
        description="test evaluation overview 1",
        labels=set(),
        metadata=dict(),
    )
