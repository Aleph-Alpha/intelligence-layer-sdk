from time import sleep
from typing import Callable, Iterable, Sequence, TypeVar
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    DefaultArgillaClient,
    Field,
    Question,
    RecordData,
)


class DummyInput(BaseModel):
    query: str


class DummyOutput(BaseModel):
    answer: str


ExpectedOutput = str


ReturnValue = TypeVar("ReturnValue")


def retry(
    f: Callable[[], ReturnValue], until: Callable[[ReturnValue], bool]
) -> ReturnValue:
    for i in range(10):
        r = f()
        if until(r):
            return r
        sleep(0.1)
    assert False, f"Condition not met after {i} retries"


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


@fixture
def qa_dataset_id(argilla_client: ArgillaClient, workspace_id: str) -> str:
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
    return argilla_client.create_dataset(workspace_id, dataset_name, fields, questions)


@fixture
def qa_records(
    argilla_client: ArgillaClient, qa_dataset_id: str
) -> Sequence[RecordData]:
    records = [
        RecordData(
            content={"question": "What is 1+1?", "answer": "2"},
            example_id="1000",
        ),
        RecordData(
            content={"question": "Wirklich?", "answer": "Ja!"},
            example_id="1001",
        ),
        RecordData(
            content={"question": "Wie ist das Wetter?", "answer": "Gut."},
            example_id="1002",
        ),
    ]
    for record in records:
        argilla_client.add_record(qa_dataset_id, record)
    return records


def test_records_returns_records_previously_added(
    argilla_client: DefaultArgillaClient,
    qa_dataset_id: str,
    qa_records: Sequence[RecordData],
) -> None:
    actual_records = argilla_client.records(qa_dataset_id)

    assert sorted(qa_records, key=lambda r: r.example_id) == sorted(
        [RecordData(**record.model_dump()) for record in actual_records],
        key=lambda r: r.example_id,
    )


def test_evaluations_returns_evaluation_results(
    argilla_client: DefaultArgillaClient,
    qa_dataset_id: str,
    qa_records: Sequence[RecordData],
) -> None:
    evaluations = [
        ArgillaEvaluation(record_id=record.id, responses={"rate-answer": 1})
        for record in argilla_client.records(qa_dataset_id)
    ]
    for evaluation in evaluations:
        argilla_client.create_evaluation(evaluation)

    actual_evaluations = retry(
        lambda: argilla_client.evaluations(qa_dataset_id),
        lambda evals: len(list(evals)) == len(evaluations),
    )

    assert sorted(actual_evaluations, key=lambda e: e.record_id) == sorted(
        evaluations, key=lambda e: e.record_id
    )
