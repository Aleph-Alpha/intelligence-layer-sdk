"""Fastapi server to run predictions."""

from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from intelligence_layer.core import ControlModel, LuminousControlModel, NoOpTracer
from intelligence_layer.examples.classify.classify import (
    ClassifyInput,
    SingleLabelClassifyOutput,
)
from intelligence_layer.examples.classify.prompt_based_classify import (
    PromptBasedClassify,
)

app = FastAPI()

load_dotenv()


def model() -> ControlModel:
    return LuminousControlModel("luminous-base-control")


@app.post("/classify")
async def classify(
    classify_input: ClassifyInput,
    luminous_control_model: LuminousControlModel = Depends(model),  # noqa: B008
) -> SingleLabelClassifyOutput:
    classify = PromptBasedClassify(luminous_control_model)
    classify_output = classify.run(classify_input, NoOpTracer())
    return classify_output
