from typing import Annotated, Iterable

from fastapi import Depends, FastAPI
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import AuthService, Task, TaskSpan


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
def client(fast_api: FastAPI) -> Iterable[TestClient]:
    with TestClient(fast_api) as client:
        yield client


@fixture
def password_auth_service() -> StubPasswordAuthService:
    return StubPasswordAuthService()
