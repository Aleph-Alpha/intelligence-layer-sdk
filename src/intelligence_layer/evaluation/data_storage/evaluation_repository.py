from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional, Sequence, cast
from uuid import uuid4

from pydantic import BaseModel

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core import JsonSerializer
from intelligence_layer.evaluation.data_storage.utils import FileBasedRepository
from intelligence_layer.evaluation.domain import (
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
    ) -> ExampleEvaluation[Evaluation]:
        if self.is_exception:
            return ExampleEvaluation(
                evaluation_id=self.evaluation_id,
                example_id=self.example_id,
                result=FailedExampleEvaluation.model_validate_json(self.json_result),
            )
        else:
            return ExampleEvaluation(
                evaluation_id=self.evaluation_id,
                example_id=self.example_id,
                result=evaluation_type.model_validate_json(self.json_result),
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

        Returns:
            :class:`Iterable` of :class:`EvaluationOverview`s.
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
    ) -> Optional[ExampleEvaluation[Evaluation]]:
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
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
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
        return [r for r in results if not isinstance(r.result, FailedExampleEvaluation)]

    def failed_example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all failed :class:`ExampleEvaluation`s for the given evaluation overview ID sorted by their example ID.

        Args:
            evaluation_id: ID of the corresponding evaluation overview.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`.

        Returns:
            A :class:`Sequence` of failed :class:`ExampleEvaluation`s.
        """
        results = self.example_evaluations(evaluation_id, evaluation_type)
        return [r for r in results if isinstance(r.result, FailedExampleEvaluation)]


class FileEvaluationRepository(EvaluationRepository, FileBasedRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in JSON files."""

    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        self.write_utf8(
            self._evaluation_overview_path(overview.id),
            overview.model_dump_json(indent=2),
        )

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        file_path = self._evaluation_overview_path(evaluation_id)
        if not file_path.exists():
            return None

        content = self.read_utf8(file_path)
        return EvaluationOverview.model_validate_json(content)

    def evaluation_overview_ids(self) -> Sequence[str]:
        overviews = (
            self.evaluation_overview(path.stem)
            for path in self._eval_root_directory().glob("*.json")
        )
        return sorted([overview.id for overview in overviews if overview is not None])

    def store_example_evaluation(
        self, example_evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        serialized_result = SerializedExampleEvaluation.from_example_result(
            example_evaluation
        )
        self.write_utf8(
            self._example_result_path(
                example_evaluation.evaluation_id, example_evaluation.example_id
            ),
            serialized_result.model_dump_json(indent=2),
        )

    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        file_path = self._example_result_path(evaluation_id, example_id)
        if not file_path.exists():
            return None

        content = self.read_utf8(file_path)
        serialized_example = SerializedExampleEvaluation.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        def load_example_evaluation_from_file_name(
            file_path: Path,
        ) -> Optional[ExampleEvaluation[Evaluation]]:
            example_id = file_path.with_suffix("").name
            return self.example_evaluation(evaluation_id, example_id, evaluation_type)

        path = self._eval_directory(evaluation_id)
        json_files = path.glob("*.json")
        example_evaluations = [
            example_result
            for example_result in (
                load_example_evaluation_from_file_name(file) for file in json_files
            )
            if example_result is not None
        ]
        return sorted(example_evaluations, key=lambda i: i.example_id)

    def _eval_root_directory(self) -> Path:
        path = self._root_directory / "evaluations"
        path.mkdir(exist_ok=True)
        return path

    def _eval_directory(self, evaluation_id: str) -> Path:
        path = self._eval_root_directory() / evaluation_id
        path.mkdir(exist_ok=True)
        return path

    def _example_result_path(self, evaluation_id: str, example_id: str) -> Path:
        return (self._eval_directory(evaluation_id) / example_id).with_suffix(".json")

    def _evaluation_overview_path(self, evaluation_id: str) -> Path:
        return self._eval_directory(evaluation_id).with_suffix(".json")


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

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        return self._evaluation_overviews.get(evaluation_id, None)

    def evaluation_overview_ids(self) -> Sequence[str]:
        return sorted(list(self._evaluation_overviews.keys()))

    def store_example_evaluation(
        self, evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        self._example_evaluations[evaluation.evaluation_id].append(evaluation)

    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        results = self.example_evaluations(evaluation_id, evaluation_type)
        filtered = (result for result in results if result.example_id == example_id)
        return next(filtered, None)

    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        example_evaluations = [
            cast(ExampleEvaluation[Evaluation], example_evaluation)
            for example_evaluation in self._example_evaluations[evaluation_id]
        ]
        return sorted(example_evaluations, key=lambda i: i.example_id)


class RecordDataSequence(BaseModel):
    records: Sequence[RecordData]


class ArgillaEvaluationRepository(EvaluationRepository):
    """Evaluation repository used in the :class:`ArgillaEvaluator`.

    Does not support storing evaluations,
    since the ArgillaEvaluator does not do automated evaluations.

    Args:
        evaluation_repository: The evaluation repository to use internally.
        argilla_client: Client to be used to connect to Argilla.
        workspace_id: The workspace ID to save the results in.
            Has to be created in Argilla beforehand.
        fields: The Argilla fields of the dataset.
        questions: The questions that will be presented to the human evaluators in Argilla.
    """

    def __init__(
        self,
        evaluation_repository: EvaluationRepository,
        argilla_client: ArgillaClient,
        workspace_id: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> None:
        super().__init__()
        self._evaluation_repository = evaluation_repository
        self._client = argilla_client
        self._workspace_id = workspace_id
        self._fields = fields
        self._questions = questions

    def initialize_evaluation(self) -> str:
        return self._client.create_dataset(
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
