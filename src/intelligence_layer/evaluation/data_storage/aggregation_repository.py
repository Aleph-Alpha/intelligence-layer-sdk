from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from wandb import Artifact, Table
from wandb.sdk.wandb_run import Run

from intelligence_layer.evaluation.data_storage.utils import FileBasedRepository
from intelligence_layer.evaluation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)


class AggregationRepository(ABC):
    @abstractmethod
    def aggregation_overview(
        self, id: str, stat_type: type[AggregatedEvaluation]
    ) -> AggregationOverview[AggregatedEvaluation] | None:
        """Returns all failed :class:`ExampleResult` instances of a given run

        Args:
            id: Identifier of the TODO
            stat_type:


        Returns:
            :class:`EvaluationOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_aggregation_overview(
        self, overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        """Stores an :class:`AggregationOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...


class FileAggregationRepository(AggregationRepository, FileBasedRepository):
    def _aggregation_root_directory(self) -> Path:
        path = self._root_directory / "aggregation"
        path.mkdir(exist_ok=True)
        return path

    def _aggregation_directory(self, eval_id: str) -> Path:
        path = self._aggregation_root_directory() / eval_id
        path.mkdir(exist_ok=True)
        return path

    def _aggregation_overview_path(self, id: str) -> Path:
        return self._aggregation_directory(id).with_suffix(".json")

    def aggregation_overview(
        self, id: str, stat_type: type[AggregatedEvaluation]
    ) -> AggregationOverview[AggregatedEvaluation] | None:
        file_path = self._aggregation_overview_path(id)
        if not file_path.exists():
            return None
        content = self.read_utf8(file_path)
        return AggregationOverview[stat_type].model_validate_json(  # type:ignore
            content
        )

    def store_aggregation_overview(
        self, overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self.write_utf8(
            self._aggregation_overview_path(overview.id),
            overview.model_dump_json(indent=2),
        )

    def aggregation_ids(self) -> Sequence[str]:
        return [path.stem for path in self._aggregation_root_directory().glob("*.json")]


class InMemoryAggregationRepository(AggregationRepository):
    def __init__(self) -> None:
        super().__init__()
        self._aggregation_overviews: dict[str, AggregationOverview[Any]] = dict()

    def aggregation_overview(
        self, id: str, stat_type: type[AggregatedEvaluation]
    ) -> AggregationOverview[AggregatedEvaluation] | None:
        return self._aggregation_overviews[id]

    def store_aggregation_overview(
        self, overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self._aggregation_overviews[overview.id] = overview


class WandbAggregationRepository(AggregationRepository):
    def __init__(self) -> None:
        super().__init__()
        self._aggregation_overviews: dict[str, Table] = dict()
        self._run: Run | None = None
        self.team_name: str = "aleph-alpha-intelligence-layer-trial"

    def aggregation_overview(
        self, id: str, stat_type: type[AggregatedEvaluation]
    ) -> AggregationOverview[AggregatedEvaluation] | None:
        table = self._get_table(id, "aggregation_overview")
        return AggregationOverview.model_validate_json(json_data=table.get_column("aggregation_overview")[0])  # type: ignore

    def store_aggregation_overview(
        self, overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self._aggregation_overviews[overview.id].add_data(  # type: ignore
            overview.model_dump_json(),
        )

    @lru_cache(maxsize=1)
    def _get_table(self, id: str, name: str) -> Table:
        if self._run is None:
            raise ValueError("Run not started")
        artifact = self._run.use_artifact(
            f"{self.team_name}/{self._run.project_name()}/{id}:latest"
        )
        return artifact.get(name)  # type: ignore

    def start_run(self, run: Run) -> None:
        self._run = run

    def finish_run(self) -> None:
        self._run = None

    def init_table(self, id: str) -> None:
        self._aggregation_overviews[id] = Table(columns=["aggregation_overview"])  # type: ignore

    def sync_table(self, id: str) -> None:
        if self._run is None:
            raise ValueError(
                "The run has not been started, are you using a WandbAggregator?"
            )
        artifact = Artifact(id, "Aggregation")
        artifact.add(self._aggregation_overviews[id], "aggregation_overview")
        self._run.log_artifact(artifact)
