from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Optional

from intelligence_layer.core.task import Input, Output
from intelligence_layer.evaluation.dataset.domain import ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.evaluation.evaluator.base_evaluator import (
    EvaluatorBase,
)


class AsyncEvaluator(EvaluatorBase[Input, Output, ExpectedOutput, Evaluation], ABC):
    @abstractmethod
    def submit(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
    ) -> PartialEvaluationOverview:
        """Submits evaluations to external service to be evaluated.

        Failed submissions are saved as FailedExampleEvaluations.

        Args:
            *run_ids: The runs to be evaluated. Each run is expected to have the same
                dataset as input (which implies their tasks have the same input-type)
                and their tasks have the same output-type. For each example in the
                dataset referenced by the runs the outputs of all runs are collected
                and if all of them were successful they are passed on to the implementation
                specific evaluation. The method compares all run of the provided ids to each other.
            num_examples: The number of examples which should be evaluated from the given runs.
                Always the first n runs stored in the evaluation repository. Defaults to None.
            abort_on_error: Abort the whole submission process if a single submission fails.
                Defaults to False.

        Returns:
            A :class:`PartialEvaluationOverview` containing submission information.
        """
        ...

    @abstractmethod
    def retrieve(self, partial_overview_id: str) -> EvaluationOverview:
        """Retrieves external evaluations and saves them to an evaluation repository.

        Failed or skipped submissions should be viewed as failed evaluations.
        Evaluations that are submitted but not yet evaluated also count as failed evaluations.

        Args:
            partial_overview_id: The id of the corresponding :class:`PartialEvaluationOverview`.

        Returns:
            An :class:`EvaluationOverview` that describes the whole evaluation.
        """
        ...


class AsyncEvaluationRepository(EvaluationRepository):
    @abstractmethod
    def store_partial_evaluation_overview(
        self, partial_evaluation_overview: PartialEvaluationOverview
    ) -> None:
        """Stores an :class:`PartialEvaluationOverview`.

        Args:
            partial_evaluation_overview: The partial overview to be persisted.
        """
        ...

    @abstractmethod
    def partial_evaluation_overview(
        self, partial_evaluation_id: str
    ) -> Optional[PartialEvaluationOverview]:
        """Returns an :class:`PartialEvaluationOverview` for the given ID.

        Args:
            partial_evaluation_id: ID of the partial evaluation overview to retrieve.

        Returns:
            :class:`PartialEvaluationOverview` if it was found, `None` otherwise.
        """
        ...

    def partial_evaluation_overviews(self) -> Iterable[PartialEvaluationOverview]:
        """Returns all :class:`PartialEvaluationOverview`s sorted by their ID.

        Yields:
            :class:`PartialEvaluationOverview`s.
        """
        for eval_id in self.partial_evaluation_overview_ids():
            evaluation_overview = self.partial_evaluation_overview(eval_id)
            if evaluation_overview is not None:
                yield evaluation_overview

    @abstractmethod
    def partial_evaluation_overview_ids(self) -> Sequence[str]:
        """Returns sorted IDs of all stored :class:`PartialEvaluationOverview`s.

        Returns:
            A :class:`Sequence` of the :class:`PartialEvaluationOverview` IDs.
        """
        ...
