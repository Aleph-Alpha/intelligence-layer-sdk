from http import HTTPStatus
from typing import Annotated, Iterable

from fastapi import Depends, FastAPI, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytest import fixture
from requests.auth import HTTPBasicAuth

from intelligence_layer.core import (
    AuthenticatedIntelligenceApp,
    AuthService,
    IntelligenceApp,
    Task,
    TaskSpan,
)


class DummyInput(BaseModel):
    text: str


class DummyOutput(BaseModel):
    response: str


class DummyTask(Task[DummyInput, DummyOutput]):
    def do_run(self, input: DummyInput, task_span: TaskSpan) -> DummyOutput:
        return DummyOutput(response=f"Response to {input.text}")


class NoOutputTask(Task[DummyInput, None]):
    def do_run(self, input: DummyInput, task_span: TaskSpan) -> None:
        return None


class DummyOutput2(BaseModel):
    number: int


class DummyTask2(Task[int, DummyOutput2]):
    def do_run(self, zzz: int, task_span: TaskSpan) -> DummyOutput2:
        return DummyOutput2(number=zzz + 1)


def config() -> str:
    return "prefix "


class TaskWithDependency(Task[DummyInput, DummyOutput]):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def do_run(self, input: DummyInput, task_span: TaskSpan) -> DummyOutput:
        return DummyOutput(response=self.prefix + input.text)


def task_with_dependency(prefix: Annotated[str, Depends(config)]) -> TaskWithDependency:
    return TaskWithDependency(prefix)


class StubPasswordAuthService(AuthService[HTTPBasicCredentials]):
    def __init__(self, expected_password: str = "password") -> None:
        self.expected_password = expected_password

    def get_permissions(
        self,
        _: frozenset[str],
        credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())],
    ) -> bool:
        if credentials.password == self.expected_password:
            return True
        else:
            return False


@fixture
def fast_api() -> FastAPI:
    return FastAPI()


@fixture
def intelligence_app(fast_api: FastAPI) -> IntelligenceApp:
    return IntelligenceApp(fast_api)


@fixture
def client(fast_api: FastAPI) -> Iterable[TestClient]:
    with TestClient(fast_api) as client:
        yield client


@fixture
def password_auth_service() -> StubPasswordAuthService:
    return StubPasswordAuthService()


def test_register_task_with_dependency(
    intelligence_app: IntelligenceApp, client: TestClient
) -> None:
    path = "/path"
    intelligence_app.register_task(task_with_dependency, DummyInput, path)
    input = "inp"
    response = client.post(path, json=DummyInput(text=input).model_dump(mode="json"))
    response.raise_for_status()

    assert DummyOutput.model_validate(response.json()).response == config() + input


def test_register_task_can_serve_multiple_tasks(
    intelligence_app: IntelligenceApp,
) -> None:
    client = TestClient(intelligence_app._fast_api_app)
    path = "/path"
    path2 = "/path2"
    intelligence_app.register_task(DummyTask(), DummyInput, path)
    intelligence_app.register_task(DummyTask2(), int, path2)

    task_input = DummyInput(text="something")
    response = client.post(path, json=task_input.model_dump(mode="json"))
    response.raise_for_status()
    task_input2 = 1
    response2 = client.post(path2, json=task_input2)
    response2.raise_for_status()

    assert task_input.text in DummyOutput.model_validate(response.json()).response
    assert DummyOutput2.model_validate(response2.json()).number == task_input2 + 1


def test_register_task_can_register_task_with_none_output(
    intelligence_app: IntelligenceApp,
) -> None:
    client = TestClient(intelligence_app._fast_api_app)
    path = "/path"
    intelligence_app.register_task(NoOutputTask(), DummyInput, path)

    response = client.post(path, json=DummyInput(text="input").model_dump(mode="json"))
    response.raise_for_status()

    assert response.text == ""
    assert response.status_code == HTTPStatus.NO_CONTENT


def test_register_task_refuses_if_incorrect_password(
    password_auth_service: StubPasswordAuthService,
) -> None:
    path = "/path"
    intelligence_app = AuthenticatedIntelligenceApp(FastAPI(), password_auth_service)
    intelligence_app.register_task(
        DummyTask(), DummyInput, path, required_permissions=frozenset({"admin"})
    )
    client = TestClient(intelligence_app._fast_api_app)
    task_input = DummyInput(text="something")
    # We know this auth service has an expected password
    auth = HTTPBasicAuth(
        "username", intelligence_app._auth_service.expected_password + "incorrect"  # type: ignore
    )

    output = client.post(path, json=task_input.model_dump(mode="json"), auth=auth)

    assert output.status_code == status.HTTP_401_UNAUTHORIZED


def test_register_task_with_dependency_refuses_if_incorrect_password(
    password_auth_service: StubPasswordAuthService,
) -> None:
    path = "/path"
    intelligence_app = AuthenticatedIntelligenceApp(FastAPI(), password_auth_service)
    intelligence_app.register_task(
        task_with_dependency,
        DummyInput,
        path,
        required_permissions=frozenset({"admin"}),
    )
    client = TestClient(intelligence_app._fast_api_app)
    auth = HTTPBasicAuth(
        "username", intelligence_app._auth_service.expected_password + "incorrect"  # type: ignore
    )

    output = client.post(
        path, json=DummyInput(text="something").model_dump(mode="json"), auth=auth
    )

    assert output.status_code == status.HTTP_401_UNAUTHORIZED


def test_register_task_if_correct_password(
    password_auth_service: StubPasswordAuthService,
) -> None:
    path = "/path"
    intelligence_app = AuthenticatedIntelligenceApp(FastAPI(), password_auth_service)
    intelligence_app.register_task(
        DummyTask(), DummyInput, path, required_permissions=frozenset({"admin"})
    )
    client = TestClient(intelligence_app._fast_api_app)
    task_input = DummyInput(text="something")
    auth = HTTPBasicAuth("username", password_auth_service.expected_password)

    output = client.post(path, json=task_input.model_dump(mode="json"), auth=auth)

    assert output.status_code == status.HTTP_200_OK
