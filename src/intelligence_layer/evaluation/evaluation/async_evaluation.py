from abc import ABC, abstractmethod
from typing import Optional

from intelligence_layer.core.task import Input, Output
from intelligence_layer.evaluation.dataset.domain import ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluator import Evaluator


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
