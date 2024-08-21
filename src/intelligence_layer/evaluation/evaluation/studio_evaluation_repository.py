from pathlib import Path

from fsspec import AbstractFileSystem # type: ignore

from intelligence_layer.connectors.base.json_serializable import JsonSerializable
from intelligence_layer.evaluation.dataset.studio_dataset_repository import StudioDatasetRepository
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
)

from intelligence_layer.evaluation.evaluation.file_evaluation_repository import FileSystemEvaluationRepository

class StudioEvaluationRepository(FileSystemEvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in a Studio Repository."""

    def __init__(self, file_system: AbstractFileSystem, root_directory: Path, evaluation_type: type[Evaluation], studio_dataset_repository: StudioDatasetRepository) -> None:
       super().__init__(file_system, root_directory)
       self.studio_dataset_repository = studio_dataset_repository
       self.evaluation_type = evaluation_type

    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        super().store_evaluation_overview(overview)

        _ = self.studio_dataset_repository.create_dataset(
            examples=self.example_evaluations(overview.id, self.evaluation_type),
            dataset_name=overview.id,
            labels=overview.labels.union(set([overview.id])),
            metadata=overview.model_dump(mode='json'),
        )