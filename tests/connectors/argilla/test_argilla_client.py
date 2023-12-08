from typing import Iterable
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors.argilla.argilla_client import (
    DefaultArgillaClient,
    Field,
    Question,
    Record,
)


class DummyInput(BaseModel):
    query: str


class DummyOutput(BaseModel):
    answer: str


ExpectedOutput = str


@fixture
def argilla_client() -> DefaultArgillaClient:
    load_dotenv()
    return DefaultArgillaClient(total_retries=8)


@fixture
def workspace_id(argilla_client: DefaultArgillaClient) -> Iterable[str]:
    try:
        workspace_id = argilla_client.create_workspace(str(uuid4()))
        yield workspace_id
    finally:
        argilla_client.delete_workspace(workspace_id)


def test_argilla_client_works(
    argilla_client: DefaultArgillaClient, workspace_id: str
) -> None:
    dataset_name = "test-dataset"
    fields = [
        Field(name="question", title="Question"),
        Field(name="answer", title="Answer"),
    ]
    questions = [
        Question(
            name="rate-answer",
            title="Rate the answer",
            description="1 means bad, 3 means amazing.",
            options=list(range(1, 4)),
        )
    ]
    dataset_id = argilla_client.create_dataset(
        workspace_id, dataset_name, fields, questions
    )

    records = [
        Record(
            content={"question": "What is 1+1?", "answer": "2"},
            example_id="1000",
        ),
        Record(
            content={"question": "Wirklich?", "answer": "Ja!"},
            example_id="1001",
        ),
        Record(
            content={"question": "Wie ist das Wetter?", "answer": "Gut."},
            example_id="1002",
        ),
    ]
    for record in records:
        argilla_client.add_record(dataset_id, record)

    loaded_records = argilla_client.records(dataset_id)
    assert sorted(records, key=lambda r: r.example_id) == sorted(
        loaded_records, key=lambda r: r.example_id
    )
