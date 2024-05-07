from abc import ABC, abstractmethod
from typing import Generic, Optional

from intelligence_layer.core.task import Input, Output
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluator import Evaluator
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput


class AsyncEvaluator(Evaluator[Input, Output, ExpectedOutput, Evaluation], ABC):
    @abstractmethod
    def submit(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
    ) -> PartialEvaluationOverview: ...

    @abstractmethod
    def retrieve(self, id: str) -> EvaluationOverview: ...


class AsyncEvaluationLogic(ABC, Generic[Input, Output, ExpectedOutput, Evaluation]):
    @abstractmethod
    def submit(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> None: ...

    @abstractmethod
    def retrieve(self, eval_id: str) -> EvaluationOverview: ...
