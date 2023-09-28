from typing import Optional
from intelligence_layer.classify import TreeNode
from pydantic import BaseModel
import matplotlib.pyplot as plt
from intelligence_layer.task import DebugLog


def test_add_debug_log_with_non_base_model() -> None:
    log = DebugLog.enabled(level="info")
    message = "Hello"
    value = ["World"]
    log.info(message, value)

    assert log.model_dump() == [{"message": message, "value": value, "level": "info"}]


def test_add_debug_log_with_base_model_is_serialized_correctly() -> None:
    class Demo(BaseModel):
        text: str

    log = DebugLog.enabled(level="info")
    message = "Hello"
    value_text = "World"
    log.info(message, Demo(text=value_text))

    assert log.model_dump() == [
        {"message": message, "value": {"text": value_text}, "level": "info"}
    ]
