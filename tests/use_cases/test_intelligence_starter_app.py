import os
from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import FastAPI, testclient
from pytest import fixture
from intelligence_layer.core.intelligence_app import IntelligenceApp
from intelligence_layer.use_cases.classify.classify import ClassifyInput


from intelligence_layer.use_cases.intelligence_starter_app import intelligence_starter_app

@fixture
def starter_app() -> IntelligenceApp:
    load_dotenv()
    aa_client = Client(os.getenv("AA_TOKEN"))
    return intelligence_starter_app(FastAPI(), aa_client)


def test_intelligence_starter_app_works(starter_app: IntelligenceApp) -> None:
    client = testclient.TestClient(starter_app._fast_api_app)

    path = "/classify"
    classify_input = ClassifyInput(chunk="chunk", labels=frozenset({"cool"}))
    response = client.post(path, json=classify_input.model_dump(mode="json"))
    response.raise_for_status()

