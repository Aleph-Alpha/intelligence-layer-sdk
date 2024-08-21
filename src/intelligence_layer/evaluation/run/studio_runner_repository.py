from pathlib import Path

from fsspec import AbstractFileSystem  # type: ignore

from intelligence_layer.core.task import Output
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)
from intelligence_layer.evaluation.run.domain import RunOverview
from intelligence_layer.evaluation.run.file_run_repository import (
    FileSystemRunRepository,
)


class StudioRunnerRepository(FileSystemRunRepository):
    """An :class:`RunRepository` that stores run results in a Studio Repository."""

    def __init__(
        self,
        file_system: AbstractFileSystem,
        root_directory: Path,
        output_type: type[Output],
        studio_dataset_repository: StudioDatasetRepository,
    ) -> None:
        super().__init__(file_system, root_directory)
        self.studio_dataset_repository = studio_dataset_repository
        self.output_type = output_type

    def store_run_overview(self, overview: RunOverview) -> None:
        super().store_run_overview(overview)

        _ = self.studio_dataset_repository.create_dataset(
            examples=self.example_outputs(overview.id, self.output_type),
            dataset_name=overview.id,
            labels=overview.labels.union(set([overview.id])),
            metadata=overview.model_dump(mode="json"),
        )
