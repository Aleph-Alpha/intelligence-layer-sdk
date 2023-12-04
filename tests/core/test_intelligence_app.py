from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytest import fixture, raises

from intelligence_layer.core import IntelligenceApp, InvalidTaskError, Task
from intelligence_layer.core.tracer import TaskSpan


class DummyInput(BaseModel):
    text: str


class DummyOutput(BaseModel):
    response: str


class DummyTask(Task[DummyInput, DummyOutput]):
    def do_run(self, input: DummyInput, task_span: TaskSpan) -> DummyOutput:
        return DummyOutput(response=f"Response to {input.text}")


class UntypedTask(Task[None, None]):
    def do_run(self, input, task_span):  # type: ignore
        return None


class TaskWithoutTaskSpanType(Task[None, None]):
    def do_run(self, input: DummyInput, task_span: str) -> DummyOutput:  # type: ignore
        return DummyOutput(response="")


class TaskWithTaskSpanAsInput(Task[TaskSpan, DummyOutput]):  # type: ignore
    def do_run(self, input: TaskSpan, task_span: TaskSpan) -> DummyOutput:
        return DummyOutput(response="")


class DummyInput2(BaseModel):
    number: int


class DummyOutput2(BaseModel):
    number: int


class DummyTask2(Task[int, DummyOutput2]):
    def do_run(self, zzz: int, task_span: TaskSpan) -> DummyOutput2:
        return DummyOutput2(number=zzz + 1)


@fixture
def intelligence_app() -> IntelligenceApp:
    return IntelligenceApp(FastAPI())


def test_serve_task_can_serve_multiple_tasks(intelligence_app: IntelligenceApp) -> None:
    client = TestClient(intelligence_app._fast_api_app)
    path = "/path"
    path2 = "/path2"
    intelligence_app.register_task(DummyTask(), path)
    intelligence_app.register_task(DummyTask2(), path2)

    task_input = DummyInput(text="something")
    response = client.post(path, json=task_input.model_dump(mode="json"))
    response.raise_for_status()
    task_input2 = 1
    response2 = client.post(path2, json=task_input2)
    response2.raise_for_status()

    assert task_input.text in DummyOutput.model_validate(response.json()).response
    assert DummyOutput2.model_validate(response2.json()).number == task_input2 + 1

def test_serve_task_refuses_if_not_authorized(
    intelligence_app: IntelligenceApp,
) -> None:
    path = "/path"

    intelligence_app.register_task(DummyTask(), path, permissions=["admin"])  
    client = TestClient(intelligence_app._fast_api_app)

    task_input = DummyInput(text="something")
    with raises(UnauthorizedException) as error:
        response = client.post(path, json=task_input.model_dump(mode="json"))
        response.raise_for_status()

def test_serve_task_throws_error_if_task_untyped(
    intelligence_app: IntelligenceApp,
) -> None:
    path = "/path"

    with raises(InvalidTaskError) as error:
        intelligence_app.register_task(UntypedTask(), path)

    assert (
        error.value.message
        == "The task `do_run` method needs a type for its input, task_span and return value."
    )


def test_serve_task_throws_error_if_no_task_span_type(
    intelligence_app: IntelligenceApp,
) -> None:
    path = "/path"

    with raises(InvalidTaskError) as error:
        intelligence_app.register_task(TaskWithoutTaskSpanType(), path)

    assert (
        error.value.message
        == "The task `do_run` method needs a `TaskSpan` type as its second argument."
    )


def test_serve_task_throws_error_if_input_is_taskspan(
    intelligence_app: IntelligenceApp,
) -> None:
    path = "/path"

    with raises(InvalidTaskError) as error:
        intelligence_app.register_task(TaskWithTaskSpanAsInput(), path)  # type: ignore
    assert (
        error.value.message
        == "The task `do_run` method cannot have a `TaskSpan` type as input."
    )
