from typing import (
    Mapping,
)

from pydantic import BaseModel
from intelligence_layer.core.task import Chunk, Probability


class ClassifyInput(BaseModel):
    """Input for a classification task.

    Attributes:
        chunk: text to be classified.
        labels: Possible labels the model will choose a label from
    """

    chunk: Chunk
    labels: frozenset[str]


class ClassifyOutput(BaseModel):
    """Output for a single label classification task.

    Attributes:
        scores: Mapping of the provided label (key) to corresponding score (value).
            The score represents how sure the model is that this is the correct label.
            This will be a value between 0 and 1.
            The sum of all probabilities will be 1.
    """

    scores: Mapping[str, Probability]
