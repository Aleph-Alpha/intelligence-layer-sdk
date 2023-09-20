from typing import Set, Sequence
from pydantic import BaseModel

from ._task import Task, DebugLog


class ClassifyInput(BaseModel):
    text: str
    """Text to be classified"""
    labels: Set[str]
    """Possible labels into which the text should be classified."""


class ClassifyOutputLabel(BaseModel):
    label: str
    """Label that the given text could have been classified as."""
    score: float
    """Confidence score of how sure the model is that this is the correct label.
    Will be a value between 0 and 1"""


class ClassifyOutput(BaseModel):
    results: Sequence[ClassifyOutputLabel]
    """Returns a score for every label provided in the input"""
    debug_log: DebugLog
    """Provides key steps, decisions, and intermediate outputs of a task's process."""


class SingleLabelClassify(Task[ClassifyInput, ClassifyOutput]):
    """Classify method for applying a single label to a given text.

    The input provides a complete set of all possible labels. The output will return a score for
    each possible label. All scores will add up to 1 and are relative to each other. The highest
    score is given to the most likely task.

    This methodology works best for classes that are easily understood, and don't require an
    explanation or examples."""

    def run(self, input: ClassifyInput) -> ClassifyOutput:
        # do python stuff / api calls
        return ClassifyOutput(
            results=[
                ClassifyOutputLabel(label=label, score=0.0) for label in input.labels
            ],
            debug_log=DebugLog(),
        )
