from pprint import pprint

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from intelligence_layer.core import Task
from intelligence_layer.core.intelligence_app import IntelligenceApp
from intelligence_layer.core.tracer import TaskSpan


class DummyInput(BaseModel):
    text: str


class DummyOutput(BaseModel):
    response: str


class DummyTask(Task[DummyInput, DummyOutput]):
    def do_run(self, input: DummyInput, task_span: TaskSpan) -> DummyOutput:
        return DummyOutput(response=f"Response to {input.text}")


class DummyInput2(BaseModel):
    number: int


class DummyOutput2(BaseModel):
    number: int


class DummyTask2(Task[int, DummyOutput2]):
    def do_run(self, zzz: int, task_span: TaskSpan) -> DummyOutput2:
        return DummyOutput2(number=zzz + 1)


def test_serve_task() -> None:
    app = IntelligenceApp(FastAPI())
    client = TestClient(app.fast_api_app)
    path = "/path"
    path2 = "/path2"
    app.register_task(DummyTask(), path)
    app.register_task(DummyTask2(), path2)

    task_input = DummyInput(text="something")
    response = client.post(path, json=task_input.model_dump(mode="json"))
    response.raise_for_status()
    task_input2 = 1  # DummyInput2(number=1)
    response2 = client.post(path2, json=task_input2)
    pprint(response2.json())
    response2.raise_for_status()

    assert task_input.text in DummyOutput.model_validate(response.json()).response
    assert DummyOutput2.model_validate(response2.json()).number == task_input2 + 1
