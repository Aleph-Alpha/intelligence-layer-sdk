from abc import ABC

from wandb.sdk.wandb_run import Run

from wandb import Artifact, Table


class WandBRepository(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._run: Run | None = None
        self._team_name: str = "aleph-alpha-intelligence-layer-trial"

    def _use_artifact(self, artifact_id: str) -> Artifact:
        if self._run is None:
            raise ValueError(
                "The run has not been started, are you using a WandbEvaluator?"
            )
        artifact: Artifact = self._run.use_artifact(
            f"{self._team_name}/{self._run.project_name()}/{artifact_id}:latest"
        )
        return artifact

    def _get_table(self, artifact: Artifact, table_name: str) -> Table:
        table = artifact.get(table_name)
        if isinstance(table, Table):
            return table
        else:
            raise ValueError(
                f"Table {table_name} not found in artifact {artifact.name}"
            )

    def start_run(self, run: Run) -> None:
        self._run = run

    def finish_run(self) -> None:
        self._run = None
