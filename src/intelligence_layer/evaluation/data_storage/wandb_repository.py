from abc import ABC

from wandb import Artifact
from wandb.sdk.wandb_run import Run


class WandBRepository(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._run: Run | None = None
        self._team_name: str = "aleph-alpha-intelligence-layer-trial"

    # @lru_cache(maxsize=2) If we want the wandb lineage to work, we cannot cache the table
    def _use_artifact(self, artifact_id: str) -> Artifact:
        if self._run is None:
            raise ValueError(
                "The run has not been started, are you using a WandbEvaluator?"
            )
        artifact: Artifact = self._run.use_artifact(
            f"{self._team_name}/{self._run.project_name()}/{artifact_id}:latest"
        )
        return artifact

    def start_run(self, run: Run) -> None:
        self._run = run

    def finish_run(self) -> None:
        self._run = None
