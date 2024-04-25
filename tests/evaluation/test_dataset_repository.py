from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

import pytest
from fsspec.implementations.memory import MemoryFileSystem  # type: ignore
from pytest import FixtureRequest, fixture, mark, raises

from intelligence_layer.evaluation import (
    DatasetRepository,
    Example,
    FileDatasetRepository,
)
from intelligence_layer.evaluation.dataset.hugging_face_dataset_repository import (
    HuggingFaceDatasetRepository,
)
from tests.conftest import DummyStringInput, DummyStringOutput


@fixture
def file_dataset_repository(tmp_path: Path) -> FileDatasetRepository:
    return FileDatasetRepository(tmp_path)


@fixture
def mocked_hugging_face_dataset_repository(
    temp_file_system: MemoryFileSystem,
) -> HuggingFaceDatasetRepository:
    class_to_patch = "intelligence_layer.evaluation.dataset.hugging_face_dataset_repository.HuggingFaceDatasetRepository"
    with patch(f"{class_to_patch}.create_repository", autospec=True), patch(
        f"{class_to_patch}.delete_repository",
        autospec=True,
    ):
        repo = HuggingFaceDatasetRepository(
            repository_id="doesn't-matter",
            token="non-existing-token",
            private=True,
        )
        repo._file_system = temp_file_system
        return repo


test_repository_fixtures = [
    "file_dataset_repository",
    "in_memory_dataset_repository",
    "mocked_hugging_face_dataset_repository",
]


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_dataset_repository_with_custom_id(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example],
        dataset_name="test-dataset",
        id="my-custom-dataset-id",
    )

    assert dataset.id == "my-custom-dataset-id"


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


@patch(target="intelligence_layer.evaluation.dataset.domain.uuid4", return_value="1234")
@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_dataset_repository_ensures_unique_dataset_ids(
    _mock_uuid4: Any,  # this is necessary as otherwise the other fixtures aren't found
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

    assert dataset_repository.dataset(dataset.id) is None
    with pytest.raises(ValueError):
        dataset_repository.examples(dataset.id, DummyStringInput, DummyStringOutput)

    dataset_repository.delete_dataset(
        dataset.id
    )  # tests whether function is idempotent


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_deleting_a_nonexistant_repo_does_not_cause_an_exception(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    dataset_repository.delete_dataset("non-existant-id")


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
def test_examples_raises_value_error_for_not_existing_dataset_id(
    repository_fixture: str, request: FixtureRequest
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    with pytest.raises(ValueError):
        dataset_repository.examples(
            "not_existing_dataset_id", DummyStringInput, type(None)
        )


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
def test_example_returns_none_for_not_existant_example_id(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )

    examples = dataset_repository.example(
        dataset.id, "not_existing_example_id", DummyStringInput, DummyStringOutput
    )

    assert examples is None


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_example_raises_error_for_not_existing_dataset_id(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    with pytest.raises(ValueError):
        dataset_repository.example(
            "not_existing_dataset_id",
            dummy_string_example.id,
            DummyStringInput,
            DummyStringOutput,
        )
    with pytest.raises(ValueError):
        dataset_repository.example(
            "not_existing_dataset_id",
            "not_existing_example_id",
            DummyStringInput,
            DummyStringOutput,
        )
