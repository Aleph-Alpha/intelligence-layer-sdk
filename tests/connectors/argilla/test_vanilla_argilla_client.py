from collections.abc import Callable, Iterable, Sequence
from time import sleep
from typing import TypeVar
from uuid import uuid4

import argilla as rg
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors import (
    ArgillaClient,
    ArgillaRatingEvaluation,
    DefaultArgillaClient,
    RecordData,
    VanillaArgillaClient,
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
    total_tries = 10
    for _ in range(total_tries):
        r = f()
        if until(r):
            return r
        sleep(0.1)
    raise AssertionError(f"Condition not met after {total_tries} retries")


@fixture
def argilla_client() -> VanillaArgillaClient:
    load_dotenv()
    return VanillaArgillaClient()


@fixture
def helper_argilla_client() -> DefaultArgillaClient:
    load_dotenv()
    return DefaultArgillaClient(total_retries=1)


@fixture
def workspace_id(argilla_client: VanillaArgillaClient) -> Iterable[str]:
    workspace = None
    try:
        workspace = argilla_client.ensure_workspace_exists(workspace_name=str(uuid4()))

        yield workspace
    finally:
        if workspace is not None:
            workspace = rg.Workspace.from_name(workspace)
            datasets = rg.list_datasets(workspace=workspace.name)
            for dataset in datasets:
                dataset.delete()
            workspace.delete()


@fixture
def qa_dataset_id(argilla_client: VanillaArgillaClient, workspace_id: str) -> str:
    dataset_name = "test-dataset"
    fields = [
        rg.TextField(name="question", title="Question"),
        rg.TextField(name="answer", title="Answer"),
    ]
    questions = [
        rg.RatingQuestion(
            name="rate-answer",
            title="Rate the answer",
            description="1 means bad, 3 means amazing.",
            values=list(range(1, 4)),
        )
    ]
    return argilla_client.ensure_dataset_exists(
        workspace_id, dataset_name, fields, questions
    )


@fixture
def qa_dataset_id_with_text_question(
    argilla_client: VanillaArgillaClient, workspace_id: str
) -> str:
    dataset_name = "test-dataset-text-question"
    fields = [
        rg.TextField(name="question", title="Question"),
        rg.TextField(name="answer", title="Answer"),
    ]
    questions = [
        rg.TextQuestion(
            name="comment-answer",
            title="Comment the answer",
            description="Just put some text in.",
            use_markdown=False,
        )
    ]
    return argilla_client.ensure_dataset_exists(
        workspace_id, dataset_name, fields, questions
    )


@pytest.mark.docker
def test_client_can_create_a_dataset(
    argilla_client: VanillaArgillaClient,
    workspace_id: str,
) -> None:
    dataset_id = argilla_client.create_dataset(
        workspace_id,
        dataset_name="name",
        fields=[rg.TextField(name="name", title="b")],
        questions=[
            rg.RatingQuestion(
                name="str", title="b", description="c", values=list(range(1, 5))
            )
        ],
    )
    datasets = rg.list_datasets(workspace_id)
    assert len(datasets) == 1
    assert str(datasets[0].id) == dataset_id


@pytest.mark.docker
def test_client_cannot_create_two_datasets_with_the_same_name(
    argilla_client: VanillaArgillaClient,
    workspace_id: str,
) -> None:
    dataset_name = str(uuid4())
    argilla_client.create_dataset(
        workspace_id,
        dataset_name=dataset_name,
        fields=[rg.TextField(name="name", title="b")],
        questions=[
            rg.RatingQuestion(
                name="str", title="b", description="c", values=list(range(1, 5))
            )
        ],
    )
    with pytest.raises(RuntimeError):
        argilla_client.create_dataset(
            workspace_id,
            dataset_name=dataset_name,
            fields=[rg.TextField(name="name", title="b")],
            questions=[
                rg.RatingQuestion(
                    name="str", title="b", description="c", values=list(range(1, 5))
                )
            ],
        )


@fixture
def qa_records(
    argilla_client: ArgillaClient, qa_dataset_id: str
) -> Sequence[RecordData]:
    records = [
        RecordData(
            content={"question": "What is 1+1?", "answer": str(i)},
            example_id=str(i),
            metadata={"model_1": "luminous-base"},
        )
        for i in range(60)
    ]
    argilla_client.add_records(qa_dataset_id, records)
    return records


@fixture
def small_qa_records(
    argilla_client: ArgillaClient, qa_dataset_id: str
) -> Sequence[RecordData]:
    records = [
        RecordData(
            content={"question": "What is 1+1?", "answer": str(i)},
            example_id=str(i),
            metadata={"model_1": "luminous-base"},
        )
        for i in range(5)
    ]
    argilla_client.add_records(qa_dataset_id, records)
    return records


@fixture
def long_qa_records(
    argilla_client: ArgillaClient, qa_dataset_id: str
) -> Sequence[RecordData]:
    records = [
        RecordData(
            content={"question": "?", "answer": str(i)},
            example_id=str(i),
            metadata={"model_1": "luminous-base"},
        )
        for i in range(1024)
    ]
    argilla_client.add_records(qa_dataset_id, records)
    return records


@pytest.mark.docker
def test_retrieving_evaluations_on_non_existant_dataset_raises_errors(
    argilla_client: VanillaArgillaClient,
) -> None:
    with pytest.raises(ValueError):
        list(argilla_client.evaluations("non_existent_dataset_id"))


@pytest.mark.docker
def test_evaluations_returns_evaluation_results(
    argilla_client: VanillaArgillaClient,
    helper_argilla_client: DefaultArgillaClient,
    qa_dataset_id: str,
    small_qa_records: Sequence[RecordData],
) -> None:
    evaluations = [
        ArgillaRatingEvaluation(
            example_id=record.example_id,
            record_id=record.id,
            responses={"rate-answer": 1},
            metadata=record.metadata,
        )
        for record in argilla_client.records(qa_dataset_id)
    ]

    for evaluation in evaluations:
        helper_argilla_client.create_evaluation(evaluation)

    res = list(argilla_client.evaluations(qa_dataset_id))
    assert len(res) == len(evaluations)


@pytest.mark.docker
def test_split_dataset_works(
    argilla_client: VanillaArgillaClient,
    qa_dataset_id: str,
    qa_records: Sequence[RecordData],
) -> None:
    n_splits = 5
    record_metadata = [
        record.metadata for record in argilla_client.records(qa_dataset_id)
    ]
    argilla_client.split_dataset(qa_dataset_id, n_splits)

    all_records = list(argilla_client.records(qa_dataset_id))
    for split in range(1, n_splits + 1):
        assert sum([record.metadata["split"] == split for record in all_records]) == 12

    new_metadata_list = [record.metadata for record in all_records]
    for old_metadata, new_metadata in zip(
        record_metadata, new_metadata_list, strict=True
    ):
        del new_metadata["split"]  # type: ignore
        assert old_metadata == new_metadata


@pytest.mark.docker
def test_split_deleting_splits_works(
    argilla_client: VanillaArgillaClient,
    qa_dataset_id: str,
    qa_records: Sequence[RecordData],
) -> None:
    n_splits = 5
    argilla_client.split_dataset(qa_dataset_id, n_splits)

    n_splits = 1
    argilla_client.split_dataset(qa_dataset_id, n_splits)

    all_records = list(argilla_client.records(qa_dataset_id))

    assert sum(["split" not in record.metadata for record in all_records]) == 60

    remote_dataset = rg.FeedbackDataset.from_argilla(id=qa_dataset_id)
    assert len(remote_dataset.metadata_properties) == 0


@pytest.mark.docker
def test_split_dataset_works_with_uneven_splits(
    argilla_client: VanillaArgillaClient,
    qa_dataset_id: str,
    qa_records: Sequence[RecordData],
) -> None:
    n_splits = 7
    argilla_client.split_dataset(qa_dataset_id, n_splits)

    all_records = list(argilla_client.records(qa_dataset_id))
    n_records_per_split = []
    for split in range(1, n_splits + 1):
        n_records_per_split.append(
            sum([record.metadata["split"] == split for record in all_records])
        )
    assert n_records_per_split == [9, 9, 9, 9, 8, 8, 8]


@pytest.mark.docker
def test_add_record_adds_multiple_records_with_same_content(
    argilla_client: VanillaArgillaClient,
    qa_dataset_id: str,
) -> None:
    first_data = RecordData(
        content={"question": "What is 1+1?", "answer": "1"},
        example_id="0",
        metadata={"first": "1", "second": "2"},
    )
    second_data = RecordData(
        content={"question": "What is 1+1?", "answer": "2"},
        example_id="0",
        metadata={"first": "2", "second": "1"},
    )

    argilla_client.add_record(qa_dataset_id, first_data)
    argilla_client.add_record(qa_dataset_id, second_data)
    assert len(list(argilla_client.records(qa_dataset_id))) == 2


@pytest.mark.docker
def test_add_record_does_not_put_example_id_into_metadata(
    argilla_client: VanillaArgillaClient,
    helper_argilla_client: DefaultArgillaClient,
    qa_dataset_id: str,
) -> None:
    first_data = RecordData(
        content={"question": "What is 1+1?", "answer": "1"},
        example_id="0",
        metadata={"first": "1", "second": "2"},
    )

    argilla_client.add_record(qa_dataset_id, first_data)

    evaluations = [
        ArgillaRatingEvaluation(
            example_id=record.example_id,
            record_id=record.id,
            responses={"rate-answer": 1},
            metadata=record.metadata,
        )
        for record in argilla_client.records(qa_dataset_id)
    ]
    helper_argilla_client.create_evaluation(evaluations[0])
    evals = list(argilla_client.evaluations(qa_dataset_id))
    assert len(evals) > 0
    assert "exampe_id" not in evals[0].metadata


@pytest.mark.docker
def test_split_dataset_can_split_long_dataset(
    argilla_client: VanillaArgillaClient,
    qa_dataset_id: str,
    long_qa_records: Sequence[RecordData],
) -> None:
    n_splits = 2
    record_metadata = [
        record.metadata for record in argilla_client.records(qa_dataset_id)
    ]
    argilla_client.split_dataset(qa_dataset_id, n_splits)

    all_records = list(argilla_client.records(qa_dataset_id))
    for split in range(1, n_splits + 1):
        assert sum([record.metadata["split"] == split for record in all_records]) == 512

    new_metadata_list = [record.metadata for record in all_records]
    for old_metadata, new_metadata in zip(
        record_metadata, new_metadata_list, strict=True
    ):
        del new_metadata["split"]  # type: ignore
        assert old_metadata == new_metadata


@pytest.mark.docker
def test_client_can_load_existing_workspace(
    argilla_client: VanillaArgillaClient,
) -> None:
    workspace_name = str(uuid4())

    created_workspace_name = argilla_client.ensure_workspace_exists(workspace_name)

    ensured_workspace_name = argilla_client.ensure_workspace_exists(workspace_name)

    assert created_workspace_name == ensured_workspace_name


@pytest.mark.docker
def test_client_can_load_existing_dataset(
    argilla_client: VanillaArgillaClient, workspace_id: str
) -> None:
    dataset_name = str(uuid4())

    created_dataset_id = argilla_client.ensure_dataset_exists(
        workspace_id=workspace_id,
        dataset_name=dataset_name,
        fields=[rg.TextField(name="a", title="b")],
        questions=[
            rg.RatingQuestion(
                name="a", title="b", description="c", values=list(range(1, 5))
            )
        ],
    )

    ensured_dataset_id = argilla_client.ensure_dataset_exists(
        workspace_id=workspace_id,
        dataset_name=dataset_name,
        fields=[rg.TextField(name="a", title="b")],
        questions=[
            rg.RatingQuestion(
                name="a", title="b", description="c", values=list(range(1, 5))
            )
        ],
    )

    assert created_dataset_id == ensured_dataset_id


@pytest.mark.docker
def test_client_can_create_a_dataset_with_text_question_records(
    argilla_client: VanillaArgillaClient, workspace_id: str
) -> None:
    dataset_id = argilla_client.create_dataset(
        workspace_id,
        dataset_name="name",
        fields=[rg.TextField(name="a", title="b")],
        questions=[
            rg.TextQuestion(name="a", title="b", description="c", use_markdown=False)
        ],
    )
    datasets = rg.list_datasets(workspace_id)
    assert len(datasets) == 1
    assert str(datasets[0].id) == dataset_id


@pytest.mark.docker
def test_add_record_to_text_question_dataset(
    argilla_client: VanillaArgillaClient,
    qa_dataset_id_with_text_question: str,
) -> None:
    first_data = RecordData(
        content={"question": "What is 1+1?", "answer": "1"},
        example_id="0",
        metadata={"first": "1", "second": "2"},
    )
    second_data = RecordData(
        content={"question": "What is 1+1?", "answer": "2"},
        example_id="0",
        metadata={"first": "2", "second": "1"},
    )

    argilla_client.add_record(qa_dataset_id_with_text_question, first_data)
    argilla_client.add_record(qa_dataset_id_with_text_question, second_data)
    assert len(list(argilla_client.records(qa_dataset_id_with_text_question))) == 2
