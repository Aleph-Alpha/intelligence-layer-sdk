from os import getenv
from pathlib import Path
from typing import Sequence, cast
from aleph_alpha_client import Client, Image
from dotenv import load_dotenv
from pytest import fixture
from intelligence_layer.retrievers.in_memory import InMemoryRetriever

from intelligence_layer.task import NoOpDebugLogger


@fixture(scope="session")
def token() -> str:
    load_dotenv()
    token = getenv("AA_TOKEN")
    assert isinstance(token, str)
    return token


@fixture(scope="session")
def client(token: str) -> Client:
    """Provide fixture for api."""
    return Client(token=token)


@fixture
def no_op_debug_logger() -> NoOpDebugLogger:
    return NoOpDebugLogger()


@fixture(scope="session")
def prompt_image() -> Image:
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    return cast(Image, Image.from_file(image_source_path))  # from_file lacks type-hint


@fixture
def in_memory_retriever(client: Client, in_memory_retriever_texts: Sequence[str]) -> InMemoryRetriever:
    return InMemoryRetriever(client, in_memory_retriever_texts, k=2)
