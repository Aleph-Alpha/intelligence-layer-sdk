from pydantic import BaseModel
from intelligence_layer.task import DebugLog


def test_add_debug_log_with_non_base_model() -> None:
    log = DebugLog()
    message = "Hello"
    value = ["World"]
    log.add(message, value)

    assert log.model_dump() == {"log": [{"message": message, "value": value}]}


def test_add_debug_log_with_base_model_is_serialized_correctly() -> None:
    class Demo(BaseModel):
        text: str

    log = DebugLog()
    message = "Hello"
    value_text = "World"
    log.add(message, Demo(text=value_text))

    assert log.model_dump() == {
        "log": [{"message": message, "value": {"text": value_text}}]
    }
