from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel

from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.evaluation.evaluator.async_evaluator import (
    AsyncEvaluationRepository,
)


class InMemoryEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in memory.

    Preferred for quick testing or to be used in Jupyter Notebooks.
    """

    def __init__(self) -> None:
        self._example_evaluations: dict[str, list[ExampleEvaluation[BaseModel]]] = (
            defaultdict(list)
        )
        self._evaluation_overviews: dict[str, EvaluationOverview] = dict()

    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        self._evaluation_overviews[overview.id] = overview
        if overview.id not in self._example_evaluations:
            self._example_evaluations[overview.id] = []

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        return self._evaluation_overviews.get(evaluation_id, None)

    def evaluation_overview_ids(self) -> Sequence[str]:
        return sorted(list(self._evaluation_overviews.keys()))

    def store_example_evaluation(
        self, evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        self._example_evaluations[evaluation.evaluation_id].append(evaluation)

    def _generate_evaluation_from_internal_evaluation(
        self,
        internal_evaluation: ExampleEvaluation[BaseModel],
        evaluation_type: type[Evaluation],
    ) -> ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]:
        if (
            type(internal_evaluation.result) is evaluation_type
            or type(internal_evaluation.result) is FailedExampleEvaluation
        ):
            return internal_evaluation  # type: ignore
        return ExampleEvaluation[Evaluation](
            evaluation_id=internal_evaluation.evaluation_id,
            example_id=internal_evaluation.example_id,
            result=evaluation_type.model_validate(internal_evaluation.result),
        )

    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[
        ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
    ]:
        results = self.example_evaluations(evaluation_id, evaluation_type)
        filtered = (result for result in results if result.example_id == example_id)
        return next(filtered, None)

    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[
        ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
    ]:
        if evaluation_id not in self._example_evaluations:
            raise ValueError(
                f"Repository does not contain an evaluation with id: {evaluation_id}"
            )

        example_evaluations = [
            self._generate_evaluation_from_internal_evaluation(
                example_evaluation, evaluation_type
            )
            for example_evaluation in self._example_evaluations[evaluation_id]
        ]
        return sorted(example_evaluations, key=lambda i: i.example_id)


class AsyncInMemoryEvaluationRepository(
    AsyncEvaluationRepository, InMemoryEvaluationRepository
):
    def __init__(self) -> None:
        super().__init__()
        self._partial_evaluation_overviews: dict[str, PartialEvaluationOverview] = (
            dict()
        )

    def store_partial_evaluation_overview(
        self, overview: PartialEvaluationOverview
    ) -> None:
        self._partial_evaluation_overviews[overview.id] = overview
        if overview.id not in self._example_evaluations:
            self._example_evaluations[overview.id] = []

    def partial_evaluation_overview(
        self, evaluation_id: str
    ) -> Optional[PartialEvaluationOverview]:
        return self._partial_evaluation_overviews.get(evaluation_id, None)

    def partial_evaluation_overview_ids(self) -> Sequence[str]:
        return sorted(list(self._partial_evaluation_overviews.keys()))
