import json
from collections.abc import Iterable

from pytest import fixture
from sqlalchemy.orm import Session

from intelligence_layer.core.tracer.tracer import JsonSerializer
from intelligence_layer.evaluation.dataset.domain import Dataset, Example
from intelligence_layer.evaluation.dataset.sql_alchemy_dataset_repository import (
    Base,
    SQLAlchemyDatasetRepository,
    SQLDataset,
    SQLExample,
)


@fixture
def sql_alchemy_dataset_repo() -> SQLAlchemyDatasetRepository:
    repo = SQLAlchemyDatasetRepository(
        url="postgresql://il_sdk:test@localhost:5433/postgres"
    )
    Base.metadata.drop_all(bind=repo.engine)
    Base.metadata.create_all(bind=repo.engine)
    return repo


@fixture
def sql_dataset_id(
    sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository,
    sequence_examples: Iterable[Example[str, None]],
) -> SQLDataset:
    return sql_alchemy_dataset_repo.create_dataset(
        examples=sequence_examples, dataset_name="my"
    ).id


def test_create_and_save_examples(
    sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository,
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    # given
    expected_ids = []
    for example in sequence_examples:
        expected_ids.append(example.id)

    # when
    sql_alchemy_dataset_repo.create_dataset(sequence_examples, "test_dataset")

    with Session(sql_alchemy_dataset_repo.engine) as session:
        sql_examples = session.query(SQLExample).all()

    # then
    for sql_example in sql_examples:
        assert sql_example.id in expected_ids


def test_dataset_ids_are_sorted_ascending(
    sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository,
) -> None:
    ids = ["123", "234", "345"]
    with Session(sql_alchemy_dataset_repo.engine) as session:
        for id in ids:
            sql_dataset = SQLDataset(
                id=id,
                name="dataset_name",
                labels=list(),
                dataset_metadata="",
                example_ids=list(),
            )
            session.add(sql_dataset)
        session.commit()

    dataset_ids = list(sql_alchemy_dataset_repo.dataset_ids())
    assert dataset_ids[0] == "123"
    assert dataset_ids[1] == "234"
    assert dataset_ids[2] == "345"


def test_dataset_exists(sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository) -> None:
    # given
    test_id = "test_id"
    with Session(sql_alchemy_dataset_repo.engine) as session:
        sql_dataset = SQLDataset(
            id=test_id,
            name="dataset_name",
            labels=list(),
            dataset_metadata={"key1": "value1", "key2": "value2"},
            example_ids=list(),
        )
        session.add(sql_dataset)
        session.commit()

    # when
    dataset = sql_alchemy_dataset_repo.dataset(test_id)

    # then
    assert dataset
    assert dataset.id == test_id


def test_dataset_does_not_exist(
    sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository,
) -> None:
    test_id = "test_id"
    non_existing_id = "wrong_id"
    with Session(sql_alchemy_dataset_repo.engine) as session:
        sql_dataset = SQLDataset(
            id=test_id,
            name="dataset_name",
            labels=list(),
            dataset_metadata="",
            example_ids=list(),
        )
        session.add(sql_dataset)
        session.commit()

    dataset = sql_alchemy_dataset_repo.dataset(non_existing_id)
    assert dataset is None


def test_fetching_from_empty_table(
    sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository,
) -> None:
    fetched_data = sql_alchemy_dataset_repo.dataset(dataset_id="123")

    assert fetched_data is None


def test_convert_sql_dataset_to_dataset() -> None:
    labels = {"first_label", "second_label"}
    metadata = {"key1": "value1", "key2": "value2"}
    id = "test_id"
    sql_dataset = SQLDataset(
        id=id,
        name="dataset_name",
        labels=list(labels) if labels else list(),
        dataset_metadata=JsonSerializer(root=metadata).model_dump(),
        example_ids=[],
    )

    dataset = sql_dataset.to_dataset()
    assert dataset.id == id
    assert dataset.metadata["key1"] == "value1"
    assert dataset.metadata["key2"] == "value2"
    assert dataset.labels == labels


def test_from_dataset_maps_sql_dataset_correctly(
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    # given
    expected_ids = []
    for example in sequence_examples:
        expected_ids.append(example.id)
    dataset = Dataset(
        id="123",
        name="dataset_name",
        labels={"label_a", "label_b"},
        metadata={"key": "value"},
    )

    # when
    mapped_sql_dataset = SQLDataset.from_dataset(
        dataset=dataset, example_ids=expected_ids
    )

    # then
    assert mapped_sql_dataset.id == dataset.id
    assert mapped_sql_dataset.name == dataset.name
    assert mapped_sql_dataset.labels == list(dataset.labels)
    assert mapped_sql_dataset.dataset_metadata is not None
    assert json.loads(mapped_sql_dataset.dataset_metadata) == {"key": "value"}
    assert mapped_sql_dataset.example_ids == expected_ids


def test_examples_returns_all_examples_for_dataset(
    sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository, sql_dataset_id: str
) -> None:
    # given

    # when

    returned_examples = sql_alchemy_dataset_repo.examples(
        dataset_id=sql_dataset_id, input_type=str, expected_output_type=None
    )

    # then
    assert len(list(returned_examples)) == 3
