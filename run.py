"""Fastapi server to run predictions."""
from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from intelligence_layer.core.model import ControlModel, LuminousControlModel
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


def model() -> ControlModel:
    return LuminousControlModel("luminous-base-control-20240215")


@app.post("/classify")
async def classify(
    classify_input: ClassifyInput,
    luminous_control_model: ControlModel = Depends(model),
) -> SingleLabelClassifyOutput:
    classify = PromptBasedClassify(luminous_control_model)
    classify_output = classify.run(classify_input, NoOpTracer())
    return classify_output
