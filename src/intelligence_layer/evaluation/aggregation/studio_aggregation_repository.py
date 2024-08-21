from pathlib import Path

from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.evaluation.aggregation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)
from intelligence_layer.evaluation.aggregation.file_aggregation_repository import FileSystemAggregationRepository
from intelligence_layer.evaluation.dataset.studio_dataset_repository import StudioDatasetRepository


class StudioAggregationRepository(FileSystemAggregationRepository):
    """An :class:`AggregationRepository` that stores aggregation results in a Studio Repository."""

    def __init__(self, file_system: LocalFileSystem, root_directory: Path, studio_dataset_repository: StudioDatasetRepository) -> None:
        super().__init__(file_system, root_directory)
        self.studio_dataset_repository = studio_dataset_repository


    def store_aggregation_overview(
        self, aggregation_overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        super().store_aggregation_overview(aggregation_overview)

        _ = self.studio_dataset_repository.create_dataset(
            examples=[aggregation_overview],
            dataset_name=aggregation_overview.id,
            labels=aggregation_overview.labels.union(set([aggregation_overview.id])),
            metadata={"aggregation_id": aggregation_overview.id},
        )