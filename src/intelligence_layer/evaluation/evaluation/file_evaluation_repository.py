from pathlib import Path
from typing import Dict, Optional, Sequence

from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
    SerializedExampleEvaluation,
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
        )

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        file_path = self._evaluation_overview_path(evaluation_id)
        if not self.exists(file_path):
            return None

        content = self.read_utf8(file_path)
        return EvaluationOverview.model_validate_json(content)

    def evaluation_overview_ids(self) -> Sequence[str]:
        return sorted(
            [
                Path(f["name"]).stem
                for f in self._file_system.ls(
                    self.path_to_str(self._eval_root_directory()), detail=True
                )
                if isinstance(f, Dict) and Path(f["name"]).suffix == ".json"
            ]
        )

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
        )

    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        file_path = self._example_result_path(evaluation_id, example_id)
        if not file_path.exists():
            return None

        content = self.read_utf8(file_path)
        serialized_example = SerializedExampleEvaluation.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        def load_example_evaluation_from_file_name(
            file_path: Path,
        ) -> Optional[ExampleEvaluation[Evaluation]]:
            example_id = file_path.with_suffix("").name
            return self.example_evaluation(evaluation_id, example_id, evaluation_type)

        path = self._eval_directory(evaluation_id)
        json_files = path.glob("*.json")
        example_evaluations = [
            example_result
            for example_result in (
                load_example_evaluation_from_file_name(file) for file in json_files
            )
            if example_result is not None
        ]
        return sorted(example_evaluations, key=lambda i: i.example_id)

    def _eval_root_directory(self) -> Path:
        path = self._root_directory / "evaluations"
        path.mkdir(exist_ok=True)
        return path

    def _eval_directory(self, evaluation_id: str) -> Path:
        path = self._eval_root_directory() / evaluation_id
        path.mkdir(exist_ok=True)
        return path

    def _example_result_path(self, evaluation_id: str, example_id: str) -> Path:
        return (self._eval_directory(evaluation_id) / example_id).with_suffix(".json")

    def _evaluation_overview_path(self, evaluation_id: str) -> Path:
        return self._eval_directory(evaluation_id).with_suffix(".json")


class FileEvaluationRepository(FileSystemEvaluationRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)

    @staticmethod
    def path_to_str(path: Path) -> str:
        return str(path)
