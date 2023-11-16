"""Fastapi server to run predictions."""
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
    LimitedConcurrencyClient,
)
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.classify.classify import (
    ClassifyInput,
    SingleLabelClassifyOutput,
)
from intelligence_layer.use_cases.classify.prompt_based_classify import (
    PromptBasedClassify,
)

app = FastAPI()

load_dotenv()


def client() -> AlephAlphaClientProtocol:
    token = os.getenv("AA_TOKEN")
    assert token is not None, "Define AA_TOKEN in your .env file"
    return LimitedConcurrencyClient.from_token(token=token)


@app.post("/classify")
async def classify(
    classify_input: ClassifyInput, client: AlephAlphaClientProtocol = Depends(client)
) -> SingleLabelClassifyOutput:
    classify = PromptBasedClassify(client)
    classify_output = classify.run(classify_input, NoOpTracer())
    return classify_output
