from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from pytest import FixtureRequest, fixture, mark, raises

from intelligence_layer.evaluation import (
    DatasetRepository,
    Example,
    FileDatasetRepository,
)
from tests.conftest import DummyStringInput, DummyStringOutput


@fixture
def file_dataset_repository(tmp_path: Path) -> FileDatasetRepository:
    return FileDatasetRepository(tmp_path)


test_repository_fixtures = [
    "file_dataset_repository",
    "in_memory_dataset_repository",
]


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_dataset_repository_can_create_and_store_a_dataset(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )
    stored_dataset = dataset_repository.dataset(dataset.id)
    stored_examples = list(
        dataset_repository.examples(
            dataset.id,
            input_type=DummyStringInput,
            expected_output_type=DummyStringOutput,
        )
    )

    assert stored_dataset == dataset
    assert len(stored_examples) == 1
    assert stored_examples[0] == dummy_string_example


@patch(
    target="intelligence_layer.evaluation.dataset.domain.uuid4", return_value="12345"
)
@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_dataset_repository_ensures_unique_dataset_ids(
    _mock_uuid4: Any,
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )
    with raises(ValueError):
        dataset_repository.create_dataset(
            examples=[dummy_string_example], dataset_name="test-dataset"
        )


@patch(
    target="intelligence_layer.evaluation.dataset.file_dataset_repository.LocalFileSystem.exists",
    return_value=True,
)
def test_file_system_dataset_repository_avoids_overriding_existing_files(
    _mock: Any,
    tmp_path: Path,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository = FileDatasetRepository(root_directory=tmp_path)

    with raises(ValueError):
        dataset_repository.create_dataset(
            examples=[dummy_string_example], dataset_name="test-dataset"
        )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_dataset_returns_none_for_not_existing_dataset_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    stored_dataset = dataset_repository.dataset("not-existing-dataset-id")

    assert stored_dataset is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_delete_dataset_deletes_a_dataset(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_examples: Iterable[Example[DummyStringInput, DummyStringOutput]],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset = dataset_repository.create_dataset(
        examples=dummy_string_examples, dataset_name="test-dataset"
    )

    dataset_repository.delete_dataset(dataset.id)

    stored_dataset = dataset_repository.dataset(dataset.id)
    examples = list(
        dataset_repository.examples(dataset.id, DummyStringInput, DummyStringOutput)
    )

    assert stored_dataset is None
    assert len(examples) == 0

    dataset_repository.delete_dataset(
        dataset.id
    )  # tests whether function is idempotent


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_datasets_returns_all_sorted_dataset(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    datasets = []
    for i in range(10):
        datasets.append(
            dataset_repository.create_dataset(
                examples=[dummy_string_example], dataset_name=f"test-dataset_{i}"
            )
        )

    stored_datasets = list(dataset_repository.datasets())

    assert stored_datasets == sorted(datasets, key=lambda dataset: dataset.id)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_dataset_ids_returns_all_sorted_ids(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_examples: Iterable[Example[DummyStringInput, DummyStringOutput]],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_ids = [
        dataset_repository.create_dataset(
            examples=dummy_string_examples, dataset_name="test-dataset"
        ).id
        for _ in range(10)
    ]

    stored_dataset_ids = dataset_repository.dataset_ids()

    assert stored_dataset_ids == sorted(dataset_ids)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_examples_returns_all_examples_sorted_by_their_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    examples = [
        Example(
            input=DummyStringInput.any(),
            expected_output=DummyStringOutput.any(),
        )
        for i in range(0, 10)
    ]
    dataset = dataset_repository.create_dataset(
        examples=examples, dataset_name="test-dataset"
    )

    stored_examples = list(
        dataset_repository.examples(dataset.id, DummyStringInput, DummyStringOutput)
    )

    assert stored_examples == sorted(examples, key=lambda example: example.id)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_examples_returns_an_empty_list_for_not_existing_dataset_id(
    repository_fixture: str, request: FixtureRequest
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    examples = list(
        dataset_repository.examples(
            "not_existing_dataset_id", DummyStringInput, type(None)
        )
    )

    assert len(examples) == 0


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_returns_example_for_existing_dataset_id_and_example_id(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )

    example = dataset_repository.example(
        dataset.id, dummy_string_example.id, DummyStringInput, DummyStringOutput
    )

    assert example == dummy_string_example


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_example_returns_none_for_not_existing_dataset_id_or_example_id(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )

    examples = [
        dataset_repository.example(
            "not_existing_dataset_id",
            dummy_string_example.id,
            DummyStringInput,
            DummyStringOutput,
        ),
        dataset_repository.example(
            dataset.id, "not_existing_example_id", DummyStringInput, DummyStringOutput
        ),
        dataset_repository.example(
            "not_existing_dataset_id",
            "not_existing_example_id",
            DummyStringInput,
            DummyStringOutput,
        ),
    ]

    assert examples == [None, None, None]
