from pydantic import BaseModel

from intelligence_layer.evaluation.evaluation.file_evaluation_repository import (
    FileEvaluationRepository,
)

"""Contains specific test for the FileEvaluationRepository. For more generic
tests, check the test_evaluation_repository file."""


class DummyType(BaseModel):
    pass


def test_evaluation_overview_ids_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert not file_evaluation_repository._eval_root_directory().exists()
    try:
        file_evaluation_repository.evaluation_overview_ids()
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()


def test_evaluation_overview_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert not file_evaluation_repository._eval_root_directory().exists()
    assert not file_evaluation_repository._eval_directory("Non-existent").exists()

    try:
        file_evaluation_repository.evaluation_overview("Non-existent")
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()


def test_example_evaluations_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert not file_evaluation_repository._eval_root_directory().exists()
    assert not file_evaluation_repository._eval_directory("Non-existent").exists()

    try:
        file_evaluation_repository.example_evaluations("Non-existent", DummyType)
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()


def test_example_evaluation_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
) -> None:
    assert not file_evaluation_repository._eval_root_directory().exists()
    assert not file_evaluation_repository._eval_directory("Non-existent").exists()

    try:
        file_evaluation_repository.example_evaluation(
            "Non-existent", "Non-existent", DummyType
        )
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()
