import os
import re
import requests
from typing import Iterable, Optional, Sequence

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


class ArgillaClient:
    def __init__(
        self, api_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        self.api_url: str = api_url or os.environ["ARGILLA_API_URL"]
        self.api_key: str = api_key or os.environ["ARGILLA_API_KEY"]
        self.headers = {
            "accept": "application/json",
            "X-Argilla-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def _create_dataset(self, name: str, workspace_id: str, guidelines: str = "No guidelines.") -> str:
        url = self.api_url + "api/v1/datasets"
        data = {
            "name": name,
            "guidelines": guidelines,
            "workspace_id": workspace_id,
            "allow_extra_metadata": True
        }
        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code // 100 == 2:
            return response.json()["id"]
        elif response.status_code == 409:
            detail = response.json()["detail"]
            pattern = r'`([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})`'
            return re.search(pattern, detail).group(1)
        response.raise_for_status()


    def upload(
        self, examples: Sequence[tuple[Example[Input, ExpectedOutput], Output]]
    ) -> None:
        name = "name_test_dataset"
        workspace_id = "15657307-9780-44e5-87ba-4ad024a49a88"
        dataset_id = self._create_dataset(name, workspace_id)
        fields = [
            
        ]

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
