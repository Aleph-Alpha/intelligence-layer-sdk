from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

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

        Returns:
            :class:`Iterable` of :class:`PartialEvaluationOverview`s.
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
