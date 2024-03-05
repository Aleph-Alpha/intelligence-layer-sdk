from pathlib import Path
from typing import Iterable

from pytest import FixtureRequest, fixture, mark

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
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_id = dataset_repository.create_dataset(examples=[dummy_string_example])
    examples = list(
        dataset_repository.examples(
            dataset_id,
            input_type=DummyStringInput,
            expected_output_type=DummyStringOutput,
        )
    )

    assert len(examples) == 1
    assert examples[0].input == dummy_string_example.input
    assert examples[0].expected_output == dummy_string_example.expected_output


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
    dataset_id = dataset_repository.create_dataset(examples=examples)

    stored_examples = list(
        dataset_repository.examples(dataset_id, DummyStringInput, DummyStringOutput)
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
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_id = dataset_repository.create_dataset(examples=[dummy_string_example])

    example = dataset_repository.example(
        dataset_id, dummy_string_example.id, DummyStringInput, DummyStringOutput
    )

    assert example == dummy_string_example


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_example_returns_none_for_not_existing_dataset_id_or_example_id(
    repository_fixture: str,
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_id = dataset_repository.create_dataset(examples=[dummy_string_example])

    examples = [
        dataset_repository.example(
            "not_existing_dataset_id",
            dummy_string_example.id,
            DummyStringInput,
            DummyStringOutput,
        ),
        dataset_repository.example(
            dataset_id, "not_existing_example_id", DummyStringInput, DummyStringOutput
        ),
        dataset_repository.example(
            "not_existing_dataset_id",
            "not_existing_example_id",
            DummyStringInput,
            DummyStringOutput,
        ),
    ]

    assert examples == [None, None, None]


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_delete_dataset_deletes_a_dataset(
    repository_fixture: str,
    dummy_string_examples: Iterable[Example[DummyStringInput, DummyStringOutput]],
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_id = dataset_repository.create_dataset(dummy_string_examples)

    dataset_repository.delete_dataset(dataset_id)

    dataset_ids = dataset_repository.dataset_ids()
    examples = list(
        dataset_repository.examples(dataset_id, DummyStringInput, DummyStringOutput)
    )

    assert dataset_id not in dataset_ids
    assert len(examples) == 0

    dataset_repository.delete_dataset(
        dataset_id
    )  # tests whether function is idempotent


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_dataset_ids_returns_all_sorted_ids(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    dataset_repository: DatasetRepository = request.getfixturevalue(repository_fixture)
    dataset_ids = []

    for i in range(0, 10):
        dataset_ids.append(
            dataset_repository.create_dataset(
                examples=[
                    Example(
                        input=DummyStringInput.any(),
                        expected_output=DummyStringOutput.any(),
                    )
                ]
            )
        )

    saved_dataset_ids = dataset_repository.dataset_ids()

    assert saved_dataset_ids == sorted(dataset_ids)
