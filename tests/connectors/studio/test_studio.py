import os
import time
from collections.abc import Generator, Sequence
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from pytest import fixture
from requests import HTTPError

from intelligence_layer.connectors import StudioClient
from intelligence_layer.core import ExportedSpan, InMemoryTracer, Task, TaskSpan


class TracerTestSubTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        task_span.log("subtask", "value")


class TracerTestTask(Task[str, str]):
    sub_task = TracerTestSubTask()

    def do_run(self, input: str, task_span: TaskSpan) -> str:
        time.sleep(0.001)
        with task_span.span("span") as sub_span:
            time.sleep(0.001)
            sub_span.log("message", "a value")
            time.sleep(0.001)
            self.sub_task.run(None, sub_span)
            time.sleep(0.001)
        self.sub_task.run(None, task_span)
        try:
            with task_span.task_span("Error task", "myInput"):
                raise ValueError("oops")
        except Exception as _:
            pass
        time.sleep(0.001)
        return "output"


@fixture
def test_trace() -> Sequence[ExportedSpan]:
    tracer = InMemoryTracer()
    task = TracerTestTask()
    task.run("my input", tracer)
    return tracer.export_for_viewing()


@fixture
def labels() -> set[str]:
    return {"label1", "label2"}


@fixture
def metadata() -> dict[str, Any]:
    return {"key": "value"}


@fixture
def studio_client() -> Generator[StudioClient]:
    client = StudioClient(project=str(uuid4()), create_project=True)
    yield client
    client._delete_project(client.project_id)


def test_cannot_connect_to_non_existing_project() -> None:
    project_name = "non-existing-project"
    with pytest.raises(ValueError, match=project_name):
        StudioClient(project="non-existing-project").project_id  # noqa: B018


def test_create_project_on_init_if_not_exists() -> None:
    project_name = str(uuid4())
    client = StudioClient(project=project_name, create_project=True)
    assert client.project_id is not None
    client._delete_project(client.project_id)


def test_can_create_two_distinct_projects_with_same_name(
    studio_client: StudioClient,
) -> None:
    project_name = str(uuid4())
    id1 = studio_client.create_project(project_name)
    id2 = studio_client.create_project(project_name)
    assert id1 != id2
    studio_client._delete_project(id1)
    studio_client._delete_project(id2)


def test_cannot_instantiate_client_with_two_projects_with_same_name(
    studio_client: StudioClient,
) -> None:
    project_name = str(uuid4())
    id1 = studio_client.create_project(project_name)
    id2 = studio_client.create_project(project_name)
    assert id1 != id2
    with pytest.raises(
        ValueError, match=f"Multiple projects with name '{project_name}' found"
    ):
        StudioClient(project=project_name)
    studio_client._delete_project(id1)
    studio_client._delete_project(id2)


def test_creating_same_projects_can_reuse_existing_project_if_told_to(
    studio_client: StudioClient,
) -> None:
    project_name = str(uuid4())
    id1 = studio_client.create_project(project_name)
    id2 = studio_client.create_project(project_name, reuse_existing=True)
    assert id1 == id2
    studio_client._delete_project(id1)


def test_delete_raises_error_when_project_not_exist() -> None:
    client = StudioClient(project="IL-default-project")
    with pytest.raises(HTTPError, match="404"):
        client._delete_project(str(uuid4()))


def test_can_upload_trace(
    test_trace: Sequence[ExportedSpan], studio_client: StudioClient
) -> None:
    id = studio_client.submit_trace(test_trace)

    assert id == str(test_trace[0].context.trace_id)


def test_cannot_upload_empty_trace(studio_client: StudioClient) -> None:
    with pytest.raises(ValueError, match="empty"):
        studio_client.submit_trace([])


def test_cannot_upload_same_trace_twice(
    test_trace: Sequence[ExportedSpan], studio_client: StudioClient
) -> None:
    studio_client.submit_trace(test_trace)
    with pytest.raises(ValueError):
        studio_client.submit_trace(test_trace)


def test_submit_trace_cannot_upload_lists_with_multiple_traces(
    studio_client: StudioClient,
) -> None:
    tracer = InMemoryTracer()
    with tracer.span("test"):
        pass
    with tracer.span("test2"):
        pass
    data = tracer.export_for_viewing()

    with pytest.raises(ValueError):
        studio_client.submit_trace(data)
    # TODO


def test_handles_invalid_url() -> None:
    with pytest.raises(ValueError, match="invalid"):
        StudioClient(str(uuid4), studio_url="unknown-url")


def test_handles_valid_but_incorrect_url() -> None:
    with pytest.raises(ValueError, match="does not point to a server"):
        StudioClient(str(uuid4), studio_url="http://invalid-test-url-123456543")


def test_handles_no_auth_configured() -> None:
    def mock_return(var: Any) -> Any:
        if var == "AA_TOKEN":
            return None
        else:
            return os.environ[var]

    with patch("os.getenv", side_effect=mock_return) as _:  # noqa: SIM117
        with pytest.raises(ValueError, match="auth_token"):
            StudioClient(str(uuid4))


def test_submit_from_tracer_can_upload_lists_with_multiple_traces(
    studio_client: StudioClient,
) -> None:
    tracer = InMemoryTracer()
    task = TracerTestTask()
    task.run("my input", tracer)
    task.run("my second input", tracer)

    id_list = set(str(span.context.trace_id) for span in tracer.export_for_viewing())

    trace_id_list = set(studio_client.submit_from_tracer(tracer))

    assert trace_id_list == id_list


def test_submit_from_tracer_works_with_empty_tracer(
    studio_client: StudioClient,
) -> None:
    tracer = InMemoryTracer()

    empty_trace_id_list = studio_client.submit_from_tracer(tracer)

    assert len(empty_trace_id_list) == 0
