from os import getenv
from pathlib import Path
from typing import cast
from aleph_alpha_client import Client, Image
from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.task import NoOpDebugLogger


@fixture(scope="session")
def client() -> Client:
    """Provide fixture for api."""
    load_dotenv()
    token = getenv("AA_TOKEN")
    assert isinstance(token, str)
    return Client(token=token)


@fixture
def no_op_debug_logger() -> NoOpDebugLogger:
    return NoOpDebugLogger()


@fixture(scope="session")
def prompt_image() -> Image:
    image_source_path = Path(__file__).parent / "dog-and-cat-cover.jpg"
    return cast(Image, Image.from_file(image_source_path))  # from_file lacks type-hint
