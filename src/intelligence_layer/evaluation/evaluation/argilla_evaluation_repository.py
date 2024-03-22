from itertools import chain
from typing import Optional, Sequence, cast
from uuid import uuid4

from pydantic import BaseModel

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)


class RecordDataSequence(BaseModel):
    records: Sequence[RecordData]


class ArgillaEvaluationRepository(EvaluationRepository):
    """Evaluation repository used in the :class:`ArgillaEvaluator` and :class:`ArgillaAggregator`.

    Only an `EvaluationOverview` is stored in the `evaluation_repository`, while the `ExampleEvaluation`s themselves are stored in Argilla.
    These `ExampleEvaluation`s are submitted to Argilla in the `store_example_evaluation` method.

    Args:
        evaluation_repository: The evaluation repository to use internally.
        argilla_client: Client to be used to connect to Argilla.
        workspace_id: The workspace ID to save the results in.
            Has to be created in Argilla beforehand.
        fields: The Argilla fields of the dataset. Has to be set for use in the :class:`ArgillaEvaluator`.
        questions: The questions that will be presented to the human evaluators in Argilla. Has to be set for use in the :class:`ArgillaEvaluator`.
    """

    def __init__(
        self,
        evaluation_repository: EvaluationRepository,
        argilla_client: ArgillaClient,
        workspace_id: str,
        fields: Optional[Sequence[Field]] = None,
        questions: Optional[Sequence[Question]] = None,
    ) -> None:
        super().__init__()
        self._evaluation_repository = evaluation_repository
        self._client = argilla_client
        self._workspace_id = workspace_id
        self._fields = fields
        self._questions = questions

    def initialize_evaluation(self) -> str:
        if self._fields is None or self._questions is None:
            raise ValueError(
                "Fields and questions have to be set to initialize the evaluation but are `None`."
            )

        return self._client.ensure_dataset_exists(
            self._workspace_id,
            str(uuid4()),
            self._fields,
            self._questions,
        )

    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        return self._evaluation_repository.store_evaluation_overview(overview)

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        return self._evaluation_repository.evaluation_overview(evaluation_id)

    def evaluation_overview_ids(self) -> Sequence[str]:
        return sorted(self._evaluation_repository.evaluation_overview_ids())

    def store_example_evaluation(
        self, evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        if isinstance(evaluation.result, RecordDataSequence):
            for record in evaluation.result.records:
                self._client.add_record(evaluation.evaluation_id, record)
        else:
            raise TypeError(
                "ArgillaEvaluationRepository does not support storing non-RecordDataSequence evaluations."
            )

    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        return self._evaluation_repository.example_evaluation(
            evaluation_id, example_id, evaluation_type
        )

    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        assert evaluation_type == ArgillaEvaluation
        successful_example_evaluations = self.successful_example_evaluations(
            evaluation_id, evaluation_type
        )
        failed_example_evaluations = self.failed_example_evaluations(
            evaluation_id, evaluation_type
        )

        return sorted(
            chain(successful_example_evaluations, failed_example_evaluations),
            key=lambda evaluation: evaluation.example_id,
        )

    def successful_example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all successfully stored :class:`ExampleEvaluation`s for the given evaluation overview ID sorted by their example ID.

        Args:
            evaluation_id: ID of the corresponding evaluation overview.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`.

        Returns:
            A :class:`Sequence` of successful :class:`ExampleEvaluation`s.
        """
        assert evaluation_type == ArgillaEvaluation
        example_evaluations = [
            ExampleEvaluation(
                evaluation_id=evaluation_id,
                example_id=e.example_id,
                # cast to Evaluation because mypy thinks ArgillaEvaluation cannot be Evaluation
                result=cast(Evaluation, e),
            )
            for e in self._client.evaluations(evaluation_id)
        ]
        return sorted(example_evaluations, key=lambda i: i.example_id)

    def failed_example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all failed :class:`ExampleEvaluation`s sorted by their example ID.

        A failed example evaluation is an :class:`ExampleEvaluation` for
        which the storage process failed, e.g., because the Argilla service
        was unresponsive.

        Args:
            evaluation_id: ID of the corresponding evaluation overview.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            A :class:`Sequence` of failed :class:`ExampleEvaluation`s.
        """
        return self._evaluation_repository.failed_example_evaluations(
            evaluation_id, evaluation_type
        )
