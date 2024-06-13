from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, SerializeAsAny

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.evaluation.evaluation.domain import (
    EvaluationFailed,
    EvaluationOverview,
)
from intelligence_layer.evaluation.run.domain import RunOverview

AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=BaseModel, covariant=True)


class AggregationOverview(BaseModel, Generic[AggregatedEvaluation], frozen=True):
    """Complete overview of the results of evaluating a :class:`Task` on a dataset.

    Created when running :meth:`Evaluator.eval_and_aggregate_runs`. Contains high-level information and statistics.

    Attributes:
        evaluation_overviews: :class:`EvaluationOverview`s used for aggregation.
        id: Aggregation overview ID.
        start: Start timestamp of the aggregation.
        end: End timestamp of the aggregation.
        end: The time when the evaluation run ended
        successful_evaluation_count: The number of examples that where successfully evaluated.
        crashed_during_evaluation_count: The number of examples that crashed during evaluation.
        failed_evaluation_count: The number of examples that crashed during evaluation
            plus the number of examples that failed to produce an output for evaluation.
        run_ids: IDs of all :class:`RunOverview`s from all linked :class:`EvaluationOverview`s.
        description: A short description.
        statistics: Aggregated statistics of the run. Whatever is returned by :meth:`Evaluator.aggregate`
        labels: Labels for filtering aggregation. Defaults to empty list.
        metadata: Additional information about the aggregation. Defaults to empty dict.

    """

    evaluation_overviews: frozenset[EvaluationOverview]
    id: str
    start: datetime
    end: datetime
    successful_evaluation_count: int
    crashed_during_evaluation_count: int
    description: str
    statistics: SerializeAsAny[AggregatedEvaluation]
    labels: set[str] = set()
    metadata: SerializableDict = dict()

    @property
    def run_ids(self) -> Sequence[str]:
        return [overview.id for overview in self.run_overviews()]

    def run_overviews(self) -> Iterable[RunOverview]:
        return set(
            run_overview
            for evaluation_overview in self.evaluation_overviews
            for run_overview in evaluation_overview.run_overviews
        )

    @property
    def failed_evaluation_count(self) -> int:
        return self.crashed_during_evaluation_count + sum(
            run_overview.failed_example_count for run_overview in self.run_overviews()
        )

    def raise_on_evaluation_failure(self) -> None:
        if self.crashed_during_evaluation_count > 0:
            raise EvaluationFailed(self.id, self.crashed_during_evaluation_count)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        res = (
            f"Aggregation Overview ID = {self.id}\n"
            f"Start time = {self.start}\n"
            f"End time = {self.end}\n"
            f"Successful example count = {self.successful_evaluation_count}\n"
            f"Count of examples crashed during evaluation = {self.failed_evaluation_count}\n"
            f'Description = "{self.description}"\n'
            f"Labels = {self.labels}\n"
            f"Metadata = {self.metadata}\n"
        )

        res += f"IDs of aggregated Evaluation Overviews = {[evaluation_overview.id for evaluation_overview in self.evaluation_overviews]}\n"
        res += f"IDs of aggregated Run Overviews = {self.run_ids}\n"

        res += "Statistics = {\n"
        res += f"{self.statistics}\n"
        res += "}\n"

        return res

    def __hash__(self) -> int:
        return hash(self.id)
