from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel

from intelligence_layer.core import JsonSerializer
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
)


class SerializedExampleEvaluation(BaseModel):
    """A json-serialized evaluation of a single example in a dataset.

    Attributes:
        evaluation_id: ID of the linked :class:`EvaluationOverview`.
        example_id: ID of the :class:`ExampleEvaluation` this evaluation was created for.
        is_exception: Will be `True` if an exception occurred during the evaluation.
        json_result: The actual serialized evaluation result.
    """

    evaluation_id: str
    example_id: str
    is_exception: bool
    json_result: str

    @classmethod
    def from_example_result(
        cls, result: ExampleEvaluation[Evaluation]
    ) -> "SerializedExampleEvaluation":
        return cls(
            evaluation_id=result.evaluation_id,
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, FailedExampleEvaluation),
            example_id=result.example_id,
        )

    def to_example_result(
        self, evaluation_type: type[Evaluation]
    ) -> ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]:
        expected_result_type = (
            FailedExampleEvaluation if self.is_exception else evaluation_type
        )
        return ExampleEvaluation(
            evaluation_id=self.evaluation_id,
            example_id=self.example_id,
            result=expected_result_type.model_validate_json(self.json_result),
        )


class EvaluationRepository(ABC):
    """Base evaluation repository interface.

    Provides methods to store and load evaluation results:
        :class:`EvaluationOverview`s and :class:`ExampleEvaluation`.
    An :class:`EvaluationOverview` is created from and is linked (by its ID)
        to multiple :class:`ExampleEvaluation`s.
    """

    def initialize_evaluation(self) -> str:
        """Initializes an :class:`EvaluationOverview` and returns its ID.

        If no extra logic is required for the initialization, this function just returns a UUID as string.
        In other cases (e.g., when a dataset has to be created in an external repository),
        this method is responsible for implementing this logic and returning the created ID.

        Returns:
            The created ID.
        """
        return str(uuid4())

    @abstractmethod
    def store_evaluation_overview(
        self, evaluation_overview: EvaluationOverview
    ) -> None:
        """Stores an :class:`EvaluationOverview`.

        Args:
            evaluation_overview: The overview to be persisted.
        """
        ...

    @abstractmethod
    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        """Returns an :class:`EvaluationOverview` for the given ID.

        Args:
            evaluation_id: ID of the evaluation overview to retrieve.

        Returns:
            :class:`EvaluationOverview` if it was found, `None` otherwise.
        """
        ...

    def evaluation_overviews(self) -> Iterable[EvaluationOverview]:
        """Returns all :class:`EvaluationOverview`s sorted by their ID.

        Yields:
            :class:`EvaluationOverview`s.
        """
        for eval_id in self.evaluation_overview_ids():
            evaluation_overview = self.evaluation_overview(eval_id)
            if evaluation_overview is not None:
                yield evaluation_overview

    @abstractmethod
    def evaluation_overview_ids(self) -> Sequence[str]:
        """Returns sorted IDs of all stored :class:`EvaluationOverview`s.

        Returns:
            A :class:`Sequence` of the :class:`EvaluationOverview` IDs.
        """
        ...

    @abstractmethod
    def store_example_evaluation(
        self, example_evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        """Stores an :class:`ExampleEvaluation`.

        Args:
            example_evaluation: The example evaluation to be persisted.
        """

    @abstractmethod
    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[
        ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
    ]:
        """Returns an :class:`ExampleEvaluation` for the given evaluation overview ID and example ID.

        Args:
            evaluation_id: ID of the linked evaluation overview.
            example_id: ID of the example evaluation to retrieve.
            evaluation_type: Type of example evaluations that the `Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            :class:`ExampleEvaluation` if it was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[
        ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
    ]:
        """Returns all :class:`ExampleEvaluation`s for the given evaluation overview ID sorted by their example ID.

        Args:
            evaluation_id: ID of the corresponding evaluation overview.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`.

        Returns:
            A :class:`Sequence` of :class:`ExampleEvaluation`s.
        """
        ...

    def successful_example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all successful :class:`ExampleEvaluation`s for the given evaluation overview ID sorted by their example ID.

        Args:
            evaluation_id: ID of the corresponding evaluation overview.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`.

        Returns:
            A :class:`Sequence` of successful :class:`ExampleEvaluation`s.
        """
        results = self.example_evaluations(evaluation_id, evaluation_type)
        return [r for r in results if not isinstance(r.result, FailedExampleEvaluation)]  # type: ignore

    def failed_example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[FailedExampleEvaluation]]:
        """Returns all failed :class:`ExampleEvaluation`s for the given evaluation overview ID sorted by their example ID.

        Args:
            evaluation_id: ID of the corresponding evaluation overview.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`.

        Returns:
            A :class:`Sequence` of failed :class:`ExampleEvaluation`s.
        """
        results = self.example_evaluations(evaluation_id, evaluation_type)
        return [r for r in results if isinstance(r.result, FailedExampleEvaluation)]  # type: ignore
