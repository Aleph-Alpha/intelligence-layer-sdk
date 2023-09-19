"""Test prediction via api."""
from pytest import fixture
from typing import Iterable
from http import HTTPStatus
from pathlib import Path
import json

from fastapi.testclient import TestClient
from run import app


@fixture
def client() -> Iterable[TestClient]:
    """Provide fixture for api."""
    try:
        yield TestClient(app)
    finally:
        pass


def test_classify(client: TestClient) -> None:
    response = client.post("/classify", json={"text": "Hello", "labels": ["yes", "no"]})
    assert response.status_code == HTTPStatus.OK
    assert response.headers.get("content-type", "") == "application/json"
    data = response.json()
