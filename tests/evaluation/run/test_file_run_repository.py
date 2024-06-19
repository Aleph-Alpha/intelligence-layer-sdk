import contextlib

import pytest
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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_example_runs_does_not_create_a_folder(
    file_run_repository: FileRunRepository,
) -> None:
    assert not file_run_repository._run_root_directory().exists()
    assert not file_run_repository._run_directory("Non-existent").exists()

    with contextlib.suppress(ValueError):
        file_run_repository.example_outputs("Non-existent", DummyType)
    assert not file_run_repository._run_root_directory().exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_example_run_does_not_create_a_folder(
    file_run_repository: FileRunRepository,
) -> None:
    assert not file_run_repository._run_root_directory().exists()
    assert not file_run_repository._run_directory("Non-existent").exists()

    with contextlib.suppress(ValueError):
        file_run_repository.example_output("Non-existent", "Non-existent", DummyType)
    assert not file_run_repository._run_root_directory().exists()
