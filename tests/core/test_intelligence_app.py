from typing import Annotated

from fastapi import Depends, FastAPI, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytest import fixture, raises
from requests.auth import HTTPBasicAuth

from intelligence_layer.core import (
    IntelligenceApp,
    RegisterTaskError,
    Task,
)
from intelligence_layer.core.intelligence_app import AuthService
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


class StubPasswordAuthService(AuthService):
    def __init__(self, expected_password: str = "password") -> None:
        self.expected_password = expected_password

    def get_permissions(
        self,
        credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())],
    ) -> frozenset[str]:
        if credentials.password == self.expected_password:
            return frozenset({"admin"})
        else:
            return frozenset({})


@fixture
def intelligence_app() -> IntelligenceApp:
    return IntelligenceApp(FastAPI())

@fixture
def password_auth_service() -> AuthService:
    return StubPasswordAuthService()


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


def test_serve_task_refuses_if_incorrect_password(
    intelligence_app: IntelligenceApp, password_auth_service: StubPasswordAuthService,
) -> None:
    path = "/path"
    intelligence_app.register_auth(password_auth_service)
    intelligence_app.register_task(
        DummyTask(), path, required_permissions=frozenset({"admin"})
    )
    client = TestClient(intelligence_app._fast_api_app)
    task_input = DummyInput(text="something")
    auth = HTTPBasicAuth("username", password_auth_service.expected_password + "incorrect")

    output = client.post(path, json=task_input.model_dump(mode="json"), auth=auth)

    assert output.status_code == status.HTTP_401_UNAUTHORIZED


def test_serve_task_if_correct_password(
    intelligence_app: IntelligenceApp, password_auth_service: StubPasswordAuthService,
) -> None:
    path = "/path"
    intelligence_app.register_auth(password_auth_service)
    intelligence_app.register_task(
        DummyTask(), path, required_permissions=frozenset({"admin"})
    )
    client = TestClient(intelligence_app._fast_api_app)
    task_input = DummyInput(text="something")
    auth = HTTPBasicAuth("username", password_auth_service.expected_password)

    output = client.post(path, json=task_input.model_dump(mode="json"), auth=auth)

    assert output.status_code == status.HTTP_200_OK


def test_serve_task_if_not_enough_permissions(
    intelligence_app: IntelligenceApp, password_auth_service: StubPasswordAuthService,
) -> None:
    path = "/path"
    intelligence_app.register_auth(password_auth_service)
    intelligence_app.register_task(
        DummyTask(), path, required_permissions=frozenset({"admin", "super"})
    )
    client = TestClient(intelligence_app._fast_api_app)
    task_input = DummyInput(text="something")
    auth = HTTPBasicAuth("username", password_auth_service.expected_password)

    output = client.post(path, json=task_input.model_dump(mode="json"), auth=auth)

    assert output.status_code == status.HTTP_401_UNAUTHORIZED


def test_serve_task_if_register_different_authentication(
    intelligence_app: IntelligenceApp, password_auth_service: StubPasswordAuthService,
) -> None:
    path = "/path"
    client = TestClient(intelligence_app._fast_api_app)
    task_input = DummyInput(text="something")
    auth_service2 = StubPasswordAuthService("otherpassword")
    auth = HTTPBasicAuth("username", password_auth_service.expected_password)

    intelligence_app.register_auth(password_auth_service)
    intelligence_app.register_task(
        DummyTask(), path, required_permissions=frozenset({"admin"})
    )

    output1 = client.post(path, json=task_input.model_dump(mode="json"), auth=auth)
    intelligence_app.register_auth(auth_service2)
    output2 = client.post(path, json=task_input.model_dump(mode="json"), auth=auth)

    assert output1.status_code == status.HTTP_200_OK
    assert output2.status_code == status.HTTP_200_OK


def test_serve_task_throws_error_if_task_untyped(
    intelligence_app: IntelligenceApp,
) -> None:
    with raises(RegisterTaskError) as error:
        intelligence_app.register_task(UntypedTask(), "/path")

    assert (
        error.value.message
        == "The task `do_run` method needs a type for its input, task_span and return value."
    )


def test_serve_task_throws_error_if_no_task_span_type(
    intelligence_app: IntelligenceApp,
) -> None:
    with raises(RegisterTaskError) as error:
        intelligence_app.register_task(TaskWithoutTaskSpanType(), "/path")

    assert (
        error.value.message
        == "The task `do_run` method needs a `TaskSpan` type as its second argument."
    )


def test_serve_task_throws_error_if_input_is_taskspan(
    intelligence_app: IntelligenceApp,
) -> None:
    with raises(RegisterTaskError) as error:
        intelligence_app.register_task(TaskWithTaskSpanAsInput(), "/path")  # type: ignore
    assert (
        error.value.message
        == "The task `do_run` method cannot have a `TaskSpan` type as input."
    )


def test_register_with_required_permissions_without_auth_throws_error(
    intelligence_app: IntelligenceApp,
) -> None:
    with raises(RegisterTaskError) as error:
        intelligence_app.register_task(DummyTask(), "/path", frozenset({"admin"}))
    assert (
        error.value.message
        == "Can't register task with required permissions without authentication registered.\nDon't forget that the order of registering tasks and authentication matters."
    )
