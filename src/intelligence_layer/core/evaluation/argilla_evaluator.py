import os
from typing import Iterable, Optional

from pydantic import BaseModel

from intelligence_layer.core.evaluation.domain import (
    AggregatedEvaluation,
    Dataset,
    Evaluation,
    Example,
    ExpectedOutput,
)
from intelligence_layer.core.evaluation.evaluator import (
    BaseEvaluator,
    EvaluationRepository,
)
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import Tracer


class ArgillaEvaluationDataset(BaseModel):
    name: str
    workspace: str


class ArgillaClient:
    def __init__(self, api_url: Optional[str], api_key: Optional[str]) -> None:
        self.api_url = api_url or os.environ["ARGILLA_API_URL"]
        self.api_key = api_key or os.environ["ARGILLA_API_KEY"]

    def upload(self, evaluation_dataset: ArgillaEvaluationDataset) -> None:
        pass


class ArgillaEvaluator(
    BaseEvaluator[Input, Output, ExpectedOutput, Evaluation, AggregatedEvaluation]
):
    def __init__(
        self,
        task: Task[Input, Output],
        repository: EvaluationRepository,
        # mapping and feedback dataset must be provided here (I think), so it can be accessed in evaluate_example
        # or: have some kind of "ArgillaRepository" that saves the feedback dataset and have it accessed via this
    ) -> None:
        super().__init__(task, repository)

    def evaluate(
        self, example: Example[Input, ExpectedOutput], eval_id: str, output: Output
    ) -> None:
        # Inserts output into feedback dataset
        # according to mapping provided
        ...

    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        # don't worry about this now
        return None  # type: ignore

    def run_dataset_and_upload(
        dataset: Dataset[Input, ExpectedOutput],
        tracer: Optional[Tracer] = None,
        # providing feedback dataset & mapping here would be preferred (as it is stateful)
    ) -> str:  # should return eval_id
        return None  # type: ignore
