from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
    SerializedExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluator.async_evaluator import (
    AsyncEvaluationRepository,
)
from intelligence_layer.evaluation.infrastructure.file_system_based_repository import (
    FileSystemBasedRepository,
)


class FileSystemEvaluationRepository(EvaluationRepository, FileSystemBasedRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in JSON files."""

    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        self.write_utf8(
            self._evaluation_overview_path(overview.id),
            overview.model_dump_json(indent=2),
            create_parents=True,
        )
        # initialize the evaluation directory to make sure the evaluation "exists", even
        # if we did not store any examples in it
        self.mkdir(self._eval_directory(overview.id))

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        file_path = self._evaluation_overview_path(evaluation_id)
        if not self.exists(file_path):
            return None

        content = self.read_utf8(file_path)
        return EvaluationOverview.model_validate_json(content)

    def evaluation_overview_ids(self) -> Sequence[str]:
        return sorted(self.file_names(self._eval_root_directory()))

    def store_example_evaluation(
        self, example_evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        serialized_result = SerializedExampleEvaluation.from_example_result(
            example_evaluation
        )
        self.write_utf8(
            self._example_result_path(
                example_evaluation.evaluation_id, example_evaluation.example_id
            ),
            serialized_result.model_dump_json(indent=2),
            create_parents=True,
        )

    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[
        ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
    ]:
        file_path = self._example_result_path(evaluation_id, example_id)
        if not self.exists(file_path.parent):
            raise ValueError(
                f"Repository does not contain an evaluation with id: {evaluation_id}"
            )
        if not self.exists(file_path):
            return None

        content = self.read_utf8(file_path)
        serialized_example = SerializedExampleEvaluation.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[
        ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
    ]:
        path = self._eval_directory(evaluation_id)
        if not self.exists(path):
            raise ValueError(
                f"Repository does not contain an evaluation with id: {evaluation_id}"
            )

        example_evaluations: list[
            ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
        ] = []
        for file_name in self.file_names(path):
            evaluation = self.example_evaluation(
                evaluation_id, file_name, evaluation_type
            )
            if evaluation is not None:
                example_evaluations.append(evaluation)

        return sorted(example_evaluations, key=lambda i: i.example_id)

    def _eval_root_directory(self) -> Path:
        path = self._root_directory / "evaluations"
        return path

    def _eval_directory(self, evaluation_id: str) -> Path:
        path = self._eval_root_directory() / evaluation_id
        return path

    def _example_result_path(self, evaluation_id: str, example_id: str) -> Path:
        path = (self._eval_directory(evaluation_id) / example_id).with_suffix(".json")
        return path

    def _evaluation_overview_path(self, evaluation_id: str) -> Path:
        path = self._eval_directory(evaluation_id).with_suffix(".json")
        return path


class FileEvaluationRepository(FileSystemEvaluationRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)

    @staticmethod
    def path_to_str(path: Path) -> str:
        return str(path)


class AsyncFileEvaluationRepository(
    FileEvaluationRepository, AsyncEvaluationRepository
):
    def store_partial_evaluation_overview(
        self, overview: PartialEvaluationOverview
    ) -> None:
        self.write_utf8(
            self._partial_evaluation_overview_path(overview.id),
            overview.model_dump_json(indent=2),
            create_parents=True,
        )
        # initialize the evaluation directory to make sure the evaluation "exists", even
        # if we did not store any examples in it
        self.mkdir(self._eval_directory(overview.id))

    def partial_evaluation_overview(
        self, evaluation_id: str
    ) -> Optional[PartialEvaluationOverview]:
        file_path = self._partial_evaluation_overview_path(evaluation_id)
        if not self.exists(file_path):
            return None

        content = self.read_utf8(file_path)
        return PartialEvaluationOverview.model_validate_json(content)

    def partial_evaluation_overview_ids(self) -> Sequence[str]:
        return sorted(self.file_names(self._eval_root_directory(), "partial_json"))

    def _partial_evaluation_overview_path(self, evaluation_id: str) -> Path:
        path = self._eval_directory(evaluation_id).with_suffix(".partial_json")
        return path
