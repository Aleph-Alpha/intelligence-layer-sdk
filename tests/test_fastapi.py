"""Test prediction via api."""
from pytest import fixture
from typing import Iterable
from http import HTTPStatus

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from run import app

load_dotenv()


@fixture
def client() -> TestClient:
    """Provide fixture for api."""
    return TestClient(app)


def test_classify(client: TestClient) -> None:
    response = client.post(
        "/classify", json={"chunk": "Hello", "labels": ["yes", "no"]}
    )
    assert response.status_code == HTTPStatus.OK
    assert response.headers.get("content-type", "") == "application/json"
    data = response.json()
    assert "scores" in data
