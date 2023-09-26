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

app = FastAPI()

load_dotenv()


def client() -> Client:
    token = os.getenv("AA_API_TOKEN")
    assert token is not None, "Define AA_API_TOKEN in your .env file"
    return Client(token=token)


@app.post("/classify")
async def classify(
    classify_input: ClassifyInput, client: Client = Depends(client)
) -> ClassifyOutput:
    classify = SingleLabelClassify(client, "info")
    classify_output = classify.run(classify_input)
    return classify_output
