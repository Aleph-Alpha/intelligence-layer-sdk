import os

from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import FastAPI, testclient
from pytest import fixture

from intelligence_layer.core import Chunk, IntelligenceApp
from intelligence_layer.use_cases.classify.classify import ClassifyInput
from intelligence_layer.use_cases.intelligence_starter_app import (
    intelligence_starter_app,
)


@fixture
def starter_app() -> IntelligenceApp:
    load_dotenv()
    aa_token = os.getenv("AA_TOKEN")
    assert aa_token
    aa_client = Client(aa_token)
    return intelligence_starter_app(FastAPI(), aa_client)


def test_intelligence_starter_app_works(starter_app: IntelligenceApp) -> None:
    client = testclient.TestClient(starter_app._fast_api_app)

    path = "/classify"
    classify_input = ClassifyInput(chunk=Chunk("chunk"), labels=frozenset({"cool"}))
    response = client.post(path, json=classify_input.model_dump(mode="json"))
    response.raise_for_status()
