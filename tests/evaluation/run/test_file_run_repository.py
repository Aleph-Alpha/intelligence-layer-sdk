import contextlib
from uuid import uuid4

from pydantic import BaseModel

from intelligence_layer.evaluation.run.file_run_repository import FileRunRepository

"""Contains specific test for the FileRunRepository. For more generic
tests, check the test_run_repository file."""


class DummyType(BaseModel):
    pass


def test_run_overview_ids_does_not_create_a_folder(
    file_run_repository: FileRunRepository,
) -> None:
    assert not file_run_repository._run_root_directory().exists()
    with contextlib.suppress(FileNotFoundError):
        file_run_repository.run_overview_ids()
    assert not file_run_repository._run_root_directory().exists()


def test_run_overview_does_not_create_a_folder(
    file_run_repository: FileRunRepository,
) -> None:
    assert not file_run_repository._run_root_directory().exists()
    assert not file_run_repository._run_directory("Non-existent").exists()

    file_run_repository.run_overview("Non-existent")
    assert not file_run_repository._run_root_directory().exists()


def test_example_runs_does_not_create_a_folder(
    file_run_repository: FileRunRepository,
) -> None:
    assert not file_run_repository._run_root_directory().exists()
    assert not file_run_repository._run_directory("Non-existent").exists()

    with contextlib.suppress(ValueError):
        file_run_repository.example_outputs("Non-existent", DummyType)
    assert not file_run_repository._run_root_directory().exists()


def test_example_run_does_not_create_a_folder(
    file_run_repository: FileRunRepository,
) -> None:
    assert not file_run_repository._run_root_directory().exists()
    assert not file_run_repository._run_directory("Non-existent").exists()

    with contextlib.suppress(ValueError):
        file_run_repository.example_output("Non-existent", "Non-existent", DummyType)
    assert not file_run_repository._run_root_directory().exists()


def test_temporary_file(
    file_run_repository: FileRunRepository,
) -> None:
    expected_run_dictionary = {
        str(uuid4()): [str(uuid4()), str(uuid4())] for _ in range(2)
    }
    # Create
    for run_id in expected_run_dictionary:
        file_run_repository.create_temporary_run_data(run_id)

    # Write
    for run_id, example_ids in expected_run_dictionary.items():
        for example_id in example_ids:
            file_run_repository.temp_store_finished_example(
                run_id=run_id, example_id=example_id
            )

    # Read
    finished_examples = file_run_repository.finished_examples()
    assert finished_examples == expected_run_dictionary

    # Delete
    run_ids = list(expected_run_dictionary.keys())
    for run_id in run_ids:
        file_run_repository.delete_temporary_run_data(run_id)
        del expected_run_dictionary[run_id]
        assert file_run_repository.finished_examples() == expected_run_dictionary
