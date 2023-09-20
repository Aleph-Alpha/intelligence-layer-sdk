"""Fastapi server to run predictions."""
import os

from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import FastAPI, Form

from intelligence_layer.classify import (
    ClassifyInput,
    ClassifyOutput,
    SingleLabelClassify,
)

app = FastAPI()
FORM = Form(...)

load_dotenv()
token = os.getenv("AA_API_TOKEN")
assert isinstance(token, str)
CLIENT = Client(token=token)


@app.post("/classify")
async def classify(classify_input: ClassifyInput) -> ClassifyOutput:
    classify = SingleLabelClassify(client=CLIENT)
    classify_output = classify.run(classify_input)
    return classify_output
