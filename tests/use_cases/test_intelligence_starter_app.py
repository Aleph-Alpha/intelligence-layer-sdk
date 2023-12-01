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
from intelligence_layer.use_cases.qa.long_context_qa import LongContextQaInput
from intelligence_layer.use_cases.summarize.summarize import LongContextSummarizeInput


@fixture
def starter_app() -> IntelligenceApp:
    load_dotenv()
    aa_token = os.getenv("AA_TOKEN")
    assert aa_token
    aa_client = Client(aa_token)
    return intelligence_starter_app(FastAPI(), aa_client)


def test_intelligence_starter_app_classify_works(starter_app: IntelligenceApp) -> None:
    client = testclient.TestClient(starter_app._fast_api_app)

    path = "/classify"
    input = ClassifyInput(chunk=Chunk("chunk"), labels=frozenset({"cool"}))
    response = client.post(path, json=input.model_dump(mode="json"))
    response.raise_for_status()


def test_intelligence_starter_app_qa_works(starter_app: IntelligenceApp) -> None:
    client = testclient.TestClient(starter_app._fast_api_app)

    path = "/qa"
    input = LongContextQaInput(text="text", question="How are you")
    response = client.post(path, json=input.model_dump(mode="json"))
    response.raise_for_status()


def test_intelligence_starter_app_summarize_works(starter_app: IntelligenceApp) -> None:
    client = testclient.TestClient(starter_app._fast_api_app)

    path = "/summarize"
    input = LongContextSummarizeInput(text="text")
    response = client.post(path, json=input.model_dump(mode="json"))
    response.raise_for_status()
