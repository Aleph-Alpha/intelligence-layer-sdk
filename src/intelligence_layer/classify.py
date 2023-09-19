from typing import Set, List, Tuple, Dict
from pydantic import BaseModel, Field

from ._output import Output, AuditTrail
from ._task import Task


class ClassifyInput(BaseModel):
    text: str = Field(description="text to be classified")
    labels: Set[str] = Field(
        description="possible labels into which the text should be classified"
    )


class ClassifyOutputLabel(BaseModel):
    label: str
    score: float


class ClassifyOutput(Output):
    results: List[ClassifyOutputLabel]


class Classify(Task):
    def definition():
        return ""

    def examples():
        return [ClassifyInput(text="This is good", labels=["positive", "negative"])]

    def run(self, classify_input: ClassifyInput):
        return ClassifyOutput(
            results=[
                ClassifyOutputLabel(label=label, score=0.0)
                for label in classify_input.labels
            ],
            audit_trail=AuditTrail(),
        )
