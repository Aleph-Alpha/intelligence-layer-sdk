"""Fastapi server to run predictions."""

from fastapi import FastAPI, status, Form, UploadFile, File, Depends, Body
from pydantic import ValidationError
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from typing import List, Union

from intelligence_layer.classify import ClassifyInput, ClassifyOutput, Classify

app = FastAPI()
FORM = Form(...)


@app.get("/classify")
async def frame_prediction():
    classify = Classify()
    return classify.as_dict()


@app.post("/classify")
async def frame_prediction(classify_input: ClassifyInput) -> ClassifyOutput:
    classify = Classify()
    classify_output = classify.run(classify_input)
    return classify_output
