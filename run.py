"""Fastapi server to run predictions."""
import os

from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form

from intelligence_layer.classify import (
    ClassifyInput,
    ClassifyOutput,
    SingleLabelClassify,
)

from intelligence_layer.task import JsonDebugLogger

app = FastAPI()

load_dotenv()


def client() -> Client:
    token = os.getenv("AA_TOKEN")
    assert token is not None, "Define AA_TOKEN in your .env file"
    return Client(token=token)


@app.post("/classify")
async def classify(
    classify_input: ClassifyInput, client: Client = Depends(client)
) -> ClassifyOutput:
    classify = SingleLabelClassify(client)
    classify_output = classify.run(classify_input, JsonDebugLogger(name="classify"))
    return classify_output
