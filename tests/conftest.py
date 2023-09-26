from os import getenv
from aleph_alpha_client import Client
from dotenv import load_dotenv
from pytest import fixture


@fixture
def client() -> Client:
    """Provide fixture for api."""
    load_dotenv()
    token = getenv("AA_API_TOKEN")
    assert isinstance(token, str)
    return Client(token=token)
