from collections.abc import Iterable, Sequence
from typing import Any
from uuid import uuid4

import argilla as rg  # type: ignore
import pytest
from argilla._exceptions import ConflictError  # type: ignore
from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.connectors import (
    ArgillaClient,
    ArgillaWrapperClient,
    RecordData,
)


@fixture
def argilla_client() -> ArgillaWrapperClient:
    load_dotenv()
    return ArgillaWrapperClient()


@fixture
def workspace_name(argilla_client: ArgillaWrapperClient) -> Iterable[str]:
    workspace_name = None
    try:
        workspace_name = argilla_client.ensure_workspace_exists(
            workspace_name=str(uuid4())
        )

        yield workspace_name
    finally:
        if workspace_name is not None:
            argilla_client._delete_workspace(workspace_name)


@fixture
def qa_dataset_id(argilla_client: ArgillaWrapperClient, workspace_name: str) -> str:
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
        workspace_name, dataset_name, fields, questions
    )


@fixture
def qa_dataset_id_with_text_question(
    argilla_client: ArgillaWrapperClient, workspace_name: str
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
        workspace_name, dataset_name, fields, questions
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
        for i in range(10)
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
def test_can_create_a_dataset(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
) -> None:
    fields = [rg.TextField(name="field", title="b")]
    questions = [
        rg.RatingQuestion(
            name="str", title="b", description="c", values=list(range(1, 5))
        )
    ]
    dataset_id = argilla_client.create_dataset(
        workspace_name,
        dataset_name="my_dataset",
        fields=fields,
        questions=questions,
    )

    dataset = argilla_client.client.datasets(
        name="my_dataset", workspace=workspace_name
    )
    assert dataset
    dataset_questions = dataset.questions

    assert dataset_questions[0].title == questions[0].title
    assert dataset_questions[0].description == questions[0].description
    assert str(dataset.id) == dataset_id


@pytest.mark.docker
def test_cannot_create_two_datasets_with_the_same_name(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
) -> None:
    dataset_name = str(uuid4())
    fields = [rg.TextField(name="field", title="b")]
    questions = [
        rg.RatingQuestion(
            name="str", title="b", description="c", values=list(range(1, 5))
        )
    ]
    argilla_client.create_dataset(
        workspace_name,
        dataset_name=dataset_name,
        fields=fields,
        questions=questions,
    )
    with pytest.raises(ConflictError):
        argilla_client.create_dataset(
            workspace_name,
            dataset_name=dataset_name,
            fields=fields,
            questions=questions,
        )


@pytest.mark.docker
def test_retrieving_evaluations_on_non_existent_dataset_raises_errors(
    argilla_client: ArgillaWrapperClient,
) -> None:
    with pytest.raises(ValueError):
        list(argilla_client.evaluations("non_existent_dataset_id"))


@pytest.mark.docker
def test_evaluations_returns_evaluation_results(
    argilla_client: ArgillaWrapperClient,
    qa_dataset_id: str,
    small_qa_records: Sequence[RecordData],
) -> None:
    records = list(argilla_client.records(qa_dataset_id))
    for record in records:
        argilla_client._create_evaluation(qa_dataset_id, record.id, {"rate-answer": 1})

    res = list(argilla_client.evaluations(qa_dataset_id))
    assert len(res) == len(records)
    assert all(evaluation.responses == {"rate-answer": 1} for evaluation in res)


@pytest.mark.docker
def test_split_dataset_works(
    argilla_client: ArgillaWrapperClient,
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
        assert sum([record.metadata["split"] == split for record in all_records]) == 2

    new_metadata_list = [record.metadata for record in all_records]
    for old_metadata, new_metadata in zip(
        record_metadata, new_metadata_list, strict=True
    ):
        del new_metadata["split"]  # type: ignore
        assert old_metadata == new_metadata


@pytest.mark.docker
def test_split_dataset_works_twice(
    argilla_client: ArgillaWrapperClient,
    qa_dataset_id: str,
    qa_records: Sequence[RecordData],
) -> None:
    n_splits = 5

    argilla_client.split_dataset(qa_dataset_id, n_splits)

    all_records = list(argilla_client.records(qa_dataset_id))
    for split in range(1, n_splits + 1):
        assert sum([record.metadata["split"] == split for record in all_records]) == 2

    argilla_client.split_dataset(qa_dataset_id, 10)
    all_records = list(argilla_client.records(qa_dataset_id))
    for split in range(1, n_splits + 1):
        assert sum([record.metadata["split"] == split for record in all_records]) == 1


@pytest.mark.docker
def test_split_deleting_splits_works(
    argilla_client: ArgillaWrapperClient,
    qa_dataset_id: str,
    qa_records: Sequence[RecordData],
) -> None:
    n_splits = 5
    argilla_client.split_dataset(qa_dataset_id, n_splits)

    n_splits = 1
    argilla_client.split_dataset(qa_dataset_id, n_splits)

    all_records = list(argilla_client.records(qa_dataset_id))

    assert sum(["split" not in record.metadata for record in all_records]) == 10

    remote_dataset = argilla_client.client.datasets(id=qa_dataset_id)
    assert remote_dataset
    assert len(remote_dataset.settings.metadata) == 0


@pytest.mark.docker
def test_split_dataset_works_with_uneven_splits(
    argilla_client: ArgillaWrapperClient,
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
    assert n_records_per_split == [2, 2, 2, 1, 1, 1, 1]


@pytest.mark.docker
def test_add_record_adds_multiple_records_with_same_content(
    argilla_client: ArgillaWrapperClient,
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
    argilla_client: ArgillaWrapperClient,
    qa_dataset_id: str,
) -> None:
    first_data = RecordData(
        content={"question": "What is 1+1?", "answer": "1"},
        example_id="0",
        metadata={"first": "1", "second": "2"},
    )

    argilla_client.add_record(qa_dataset_id, first_data)
    record = next(iter(argilla_client.records(qa_dataset_id)))

    argilla_client._create_evaluation(
        dataset_id=qa_dataset_id, record_id=record.id, data={"rate-answer": 1}
    )
    evals = list(argilla_client.evaluations(qa_dataset_id))
    assert len(evals) > 0
    assert "exampe_id" not in evals[0].metadata


@pytest.mark.docker
def test_split_dataset_can_split_long_dataset(
    argilla_client: ArgillaWrapperClient,
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
    argilla_client: ArgillaWrapperClient,
) -> None:
    workspace_name = str(uuid4())

    created_workspace_name = argilla_client.ensure_workspace_exists(workspace_name)

    ensured_workspace_name = argilla_client.ensure_workspace_exists(workspace_name)

    assert created_workspace_name == ensured_workspace_name


@pytest.mark.docker
def test_client_can_load_existing_dataset(
    argilla_client: ArgillaWrapperClient, workspace_name: str
) -> None:
    dataset_name = str(uuid4())

    created_dataset_id = argilla_client.ensure_dataset_exists(
        workspace_name=workspace_name,
        dataset_name=dataset_name,
        fields=[rg.TextField(name="text", title="b")],
        questions=[
            rg.RatingQuestion(
                name="question", title="b", description="c", values=list(range(1, 5))
            )
        ],
    )

    ensured_dataset_id = argilla_client.ensure_dataset_exists(
        workspace_name=workspace_name,
        dataset_name=dataset_name,
        fields=[rg.TextField(name="c", title="b")],
        questions=[
            rg.RatingQuestion(
                name="d", title="b", description="c", values=list(range(1, 5))
            )
        ],
    )

    assert created_dataset_id == ensured_dataset_id


def assert_question_works(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
    question: Any,
    question_data: dict[str, Any],
) -> None:
    # given
    fields = [
        rg.TextField(name="field"),
    ]
    dataset_id = argilla_client.create_dataset(
        workspace_name=workspace_name,
        dataset_name="question-tests",
        fields=fields,
        questions=[question],
    )
    example_id = "id"
    # when
    argilla_client.add_record(
        dataset_id,
        RecordData(content={"field": "Test content."}, example_id=example_id),
    )
    record = next(iter(argilla_client.records(dataset_id)))
    argilla_client._create_evaluation(
        dataset_id=dataset_id, record_id=record.id, data=question_data
    )
    results = list(argilla_client.evaluations(dataset_id=dataset_id))
    # then

    assert len(results) == 1


def test_text_question_works(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
):
    assert_question_works(
        argilla_client,
        workspace_name,
        rg.TextQuestion(name="text"),
        {"text": "some text"},
    )


def test_label_question_works(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
):
    assert_question_works(
        argilla_client,
        workspace_name,
        rg.LabelQuestion(
            name="label",
            labels={"YES": "Shown Yes", "NO": "Shown No"},
            visible_labels=None,
        ),
        {"label": "YES"},
    )


def test_multi_label_question_works(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
):
    assert_question_works(
        argilla_client,
        workspace_name,
        rg.MultiLabelQuestion(
            name="multilabel",
            labels={"a": "Shown A", "b": "Shown B"},
            visible_labels=None,
        ),
        {"multilabel": ["a", "b"]},
    )


def test_rating_question_works(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
):
    assert_question_works(
        argilla_client,
        workspace_name,
        rg.RatingQuestion(name="rating", values=[1, 3, 5]),
        {"rating": 3},
    )


def test_span_question_works(
    argilla_client: ArgillaWrapperClient,
    workspace_name: str,
):
    assert_question_works(
        argilla_client,
        workspace_name,
        rg.SpanQuestion(
            name="test",
            field="field",
            labels=["a", "b", "c"],
            allow_overlapping=False,
            visible_labels=None,
        ),
        {
            "test": [
                {
                    "start": 0,
                    "end": 3,
                    "label": "a",
                }
            ]
        },
    )
