from intelligence_layer.evaluation.evaluation.file_evaluation_repository import (
    FileEvaluationRepository,
)
"""Contains specific test for the FileEvaluationRepository. For more generic
tests, check the test_evaluation_repository file."""

def test_evaluation_overview_ids_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
):
    assert not file_evaluation_repository._eval_root_directory().exists()
    try:
        file_evaluation_repository.evaluation_overview_ids()
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()


def test_evaluation_overview_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
):
    assert not file_evaluation_repository._eval_root_directory().exists()
    assert not file_evaluation_repository._eval_directory("Non-existant").exists()

    try:
        file_evaluation_repository.evaluation_overview("Non-existant")
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()


def test_example_evaluations_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
):
    assert not file_evaluation_repository._eval_root_directory().exists()
    assert not file_evaluation_repository._eval_directory("Non-existant").exists()

    try:
        file_evaluation_repository.example_evaluations("Non-existant", None)
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()


def test_example_evaluation_does_not_create_a_folder(
    file_evaluation_repository: FileEvaluationRepository,
):
    assert not file_evaluation_repository._eval_root_directory().exists()
    assert not file_evaluation_repository._eval_directory("Non-existant").exists()

    try:
        file_evaluation_repository.example_evaluation(
            "Non-existant", "Non-existant", None
        )
    except Exception:
        pass
    assert not file_evaluation_repository._eval_root_directory().exists()
