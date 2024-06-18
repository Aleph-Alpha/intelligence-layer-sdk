import contextlib
from uuid import uuid4

from pydantic import BaseModel

from intelligence_layer.evaluation.run.file_run_repository import FileRunRepository
from intelligence_layer.evaluation.run.run_repository import RecoveryData

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
    test_hash = str(uuid4())
    run_id = str(uuid4())

    expected_recovery_data = RecoveryData(
        run_id=run_id, finished_examples=[str(uuid4()), str(uuid4())]
    )

    # Create
    file_run_repository.create_temporary_run_data(test_hash, run_id)

    # Write
    for example_id in expected_recovery_data.finished_examples:
        file_run_repository.temp_store_finished_example(
            tmp_hash=test_hash, example_id=example_id
        )

    # Read
    finished_examples = file_run_repository.finished_examples(test_hash)
    assert finished_examples == expected_recovery_data

    # Delete
    file_run_repository.delete_temporary_run_data(test_hash)
    assert file_run_repository.finished_examples(test_hash) is None
