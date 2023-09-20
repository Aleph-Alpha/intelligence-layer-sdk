"""Fastapi server to run predictions."""

from fastapi import FastAPI, Form

from intelligence_layer.classify import (
    ClassifyInput,
    ClassifyOutput,
    SingleLabelClassify,
)

app = FastAPI()
FORM = Form(...)


@app.post("/classify")
async def classify(classify_input: ClassifyInput) -> ClassifyOutput:
    classify = SingleLabelClassify()
    classify_output = classify.run(classify_input)
    return classify_output
