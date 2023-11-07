"""Fastapi server to run predictions."""
import os

from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.classify.classify import ClassifyInput, ClassifyOutput
from intelligence_layer.use_cases.classify.single_label_classify import (
    SingleLabelClassify,
)

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
    classify_output = classify.run(classify_input, NoOpTracer())
    return classify_output
