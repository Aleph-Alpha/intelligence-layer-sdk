from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from fsspec.implementations.memory import MemoryFileSystem  # type: ignore
from pydantic import ValidationError
from pytest import FixtureRequest, fixture, mark, raises

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import Input
from intelligence_layer.evaluation import (
    DatasetRepository,
    Example,
    ExpectedOutput,
    FileDatasetRepository,
)
from intelligence_layer.evaluation.dataset.hugging_face_dataset_repository import (
    HuggingFaceDatasetRepository,
)
from tests.evaluation.conftest import (
    DummyStringExpectedOutput,
    DummyStringInput,
    DummyStringOutput,
)


@fixture
def file_dataset_repository(tmp_path: Path) -> FileDatasetRepository:
    return FileDatasetRepository(tmp_path)


@fixture
def mocked_hugging_face_dataset_repository(
    temp_file_system: MemoryFileSystem,
) -> HuggingFaceDatasetRepository:
    class_to_patch = "intelligence_layer.evaluation.dataset.hugging_face_dataset_repository.HuggingFaceDatasetRepository"
    with (
        patch(f"{class_to_patch}.create_repository", autospec=True),
        patch(
            f"{class_to_patch}.delete_repository",
            autospec=True,
        ),
    ):
        repo = HuggingFaceDatasetRepository(
            repository_id="doesn't-matter",
            token="non-existing-token",
            private=True,
        )
        repo._file_system = temp_file_system
        return repo


def get_example_via_both_retrieval_methods(
    dataset_repository: DatasetRepository,
    dataset_id: str,
    example_id: str,
    input_type: type[Input],
    expected_output_type: type[ExpectedOutput],
) -> Iterable[Example[Input, ExpectedOutput] | None]:
    yield dataset_repository.example(
        dataset_id, example_id, input_type, expected_output_type
    )
    yield next(
        iter(dataset_repository.examples(dataset_id, input_type, expected_output_type))
    )


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
def test_dataset_repository_create_dataset_sets_default_values(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )

    assert dataset.id is not None
    assert dataset.name == "test-dataset"
    assert dataset.labels == set()
    assert dataset.metadata == dict()


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_create_dataset_explicit_values_overwrite_defaults(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
) -> None:
    expected_id = str(uuid4())
    expected_name = "test_name"
    expected_labels = {"test_label"}
    expected_metadata: SerializableDict = dict({"test_key": "test_value"})

    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)

    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example],
        dataset_name=expected_name,
        id=expected_id,
        labels=expected_labels,
        metadata=expected_metadata,
    )

    assert dataset.id == expected_id
    assert dataset.name == expected_name
    assert dataset.labels == expected_labels
    assert dataset.metadata == expected_metadata


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_can_create_and_store_a_dataset(
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
            expected_output_type=DummyStringExpectedOutput,
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
def test_ensures_unique_dataset_ids(
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
            input=DummyStringInput(),
            expected_output=DummyStringOutput(),
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
def test_examples_skips_blacklisted_examples(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    examples = [
        Example(
            input=DummyStringInput(),
            expected_output=DummyStringOutput(),
        )
        for _ in range(0, 10)
    ]
    examples_to_skip = frozenset(example.id for example in examples[2:5])
    dataset = dataset_repository.create_dataset(
        examples=examples, dataset_name="test-dataset"
    )

    retrieved_examples = list(
        dataset_repository.examples(
            dataset.id, DummyStringInput, DummyStringOutput, examples_to_skip
        )
    )

    assert len(retrieved_examples) == len(examples) - len(examples_to_skip)
    assert all(example.id not in examples_to_skip for example in retrieved_examples)


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
            "not_existing_dataset_id", DummyStringInput, DummyStringExpectedOutput
        )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_returns_example_for_existing_dataset_id_and_example_id(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringExpectedOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )

    example = dataset_repository.example(
        dataset.id, dummy_string_example.id, DummyStringInput, DummyStringExpectedOutput
    )

    assert example == dummy_string_example


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_example_returns_none_for_not_existant_example_id(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringExpectedOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )

    examples = dataset_repository.example(
        dataset.id,
        "not_existing_example_id",
        DummyStringInput,
        DummyStringExpectedOutput,
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


def assert_dataset_examples_works_with_type(
    dataset_repository: DatasetRepository, value: Any, type: type[Input]
) -> None:
    expected_example = Example(
        input=value,
        expected_output=value,
        metadata=None,
    )
    dataset = dataset_repository.create_dataset(
        examples=[expected_example], dataset_name="test-dataset"
    )
    for example in get_example_via_both_retrieval_methods(
        dataset_repository,
        dataset.id,
        expected_example.id,
        type,
        type,
    ):
        assert example == expected_example


@mark.parametrize("repository_fixture", test_repository_fixtures)
@mark.parametrize(
    "value, value_type", [(1, int), ("1", str), (None, type(None)), (False, bool)]
)
def test_retrieving_with_int_types_works(
    repository_fixture: str,
    value: Any,
    value_type: type,
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    assert_dataset_examples_works_with_type(dataset_repository, value, value_type)


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_example_creating_with_json_and_reading_with_actual_type_works(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringExpectedOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    json_example = Example(
        input=dummy_string_example.input.model_dump(),
        expected_output=dummy_string_example.expected_output.model_dump(),
        metadata=dummy_string_example.metadata,
    )
    dataset = dataset_repository.create_dataset(
        examples=[json_example], dataset_name="test-dataset"
    )

    for new_example in get_example_via_both_retrieval_methods(
        dataset_repository,
        dataset.id,
        json_example.id,
        DummyStringInput,
        DummyStringExpectedOutput,
    ):
        assert new_example is not None
        assert new_example.input == dummy_string_example.input
        assert new_example.expected_output == dummy_string_example.expected_output
        assert new_example.metadata == dummy_string_example.metadata


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_example_retrieving_with_wrong_types_gives_error(
    repository_fixture: str,
    request: FixtureRequest,
    dummy_string_example: Example[DummyStringInput, DummyStringExpectedOutput],
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset = dataset_repository.create_dataset(
        examples=[dummy_string_example], dataset_name="test-dataset"
    )
    with pytest.raises(ValidationError):
        dataset_repository.example(dataset.id, dummy_string_example.id, int, int)

    with pytest.raises(ValidationError):
        next(iter(dataset_repository.examples(dataset.id, int, int)))
