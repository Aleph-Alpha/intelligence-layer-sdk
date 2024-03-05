from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
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
        evaluation_id: Identifier of the run the evaluated example belongs to.
        example_id: Unique identifier of the example this evaluation was created for.
        is_exception: Will be `True` if an exception occurred during evaluation.
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

    Provides methods to store and load results of an evaluation run (evaluation overview).
    """

    def create_evaluation_dataset(self) -> str:
        """Generates an ID for the dataset and creates it if necessary.

        If no extra logic is required to create the dataset for the run,
        this function just returns a UUID as string.
        In other cases (like when the dataset has to be created in an external repository),
        this method is responsible for implementing this logic and returning the ID.

        Returns:
            The ID of the dataset.
        """
        return str(uuid4())

    @abstractmethod
    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        """Stores an :class:`EvaluationOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...

    @abstractmethod
    def evaluation_overview(self, evaluation_id: str) -> EvaluationOverview | None:
        """Returns an :class:`EvaluationOverview` of a given run by its id.

        Args:
            evaluation_id: Identifier of the eval run to obtain the overview for.

        Returns:
            :class:`EvaluationOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def evaluation_overview_ids(self) -> Sequence[str]:
        """Returns IDs of all stored evaluation overviews.

        Returns:
            The IDs of all stored evaluation overviews.
        """
        ...

    @abstractmethod
    def store_example_evaluation(self, result: ExampleEvaluation[Evaluation]) -> None:
        """Stores an :class:`ExampleEvaluation` in the repository.

        Args:
            result: The result to be persisted.
        """

    @abstractmethod
    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        """Returns an :class:`ExampleEvaluation` for the given evaluation overview and example ID.

        Args:
            evaluation_id: ID of the evaluation overview to obtain the results for.
            example_id: Example ID: will match :class:`ExampleEvaluation` ID.
            evaluation_type: Type of evaluations that the `Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            :class:`ExampleEvaluation` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all :class:`ExampleEvaluation`s for the given evaluation overview ID.

        Args:
            evaluation_id: ID of the evaluation overview to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`.

        Returns:
            Sorted sequence of all :class:`ExampleEvaluations`.
        """
        ...

    def failed_example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all failed :class:`ExampleEvaluation`s for the given evaluation overview ID sorted by their example ID.

        Args:
            evaluation_id: Identifier of the evaluation run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            Sorted sequence of all :class:`ExampleEvaluations`.
        """
        results = self.example_evaluations(evaluation_id, evaluation_type)
        return [r for r in results if isinstance(r.result, FailedExampleEvaluation)]


class FileEvaluationRepository(EvaluationRepository, FileBasedRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in json-files.

    Args:
        root_directory: The folder where the json-files are stored. The folder (along with its parents)
            will be created if it does not exist yet.
    """

    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        self.write_utf8(
            self._evaluation_run_overview_path(overview.id),
            overview.model_dump_json(indent=2),
        )

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        file_path = self._evaluation_run_overview_path(evaluation_id)
        if not file_path.exists():
            return None
        content = self.read_utf8(file_path)
        return EvaluationOverview.model_validate_json(content)

    def evaluation_overview_ids(self) -> Sequence[str]:
        overviews = (
            self.evaluation_overview(path.stem)
            for path in self._eval_root_directory().glob("*.json")
        )
        return [overview.id for overview in overviews if overview is not None]

    def store_example_evaluation(self, result: ExampleEvaluation[Evaluation]) -> None:
        serialized_result = SerializedExampleEvaluation.from_example_result(result)
        self.write_utf8(
            self._example_result_path(result.evaluation_id, result.example_id),
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
        def fetch_result_from_file_name(
            path: Path,
        ) -> Optional[ExampleEvaluation[Evaluation]]:
            id = path.with_suffix("").name
            return self.example_evaluation(evaluation_id, id, evaluation_type)

        path = self._eval_directory(evaluation_id)
        logs = path.glob("*.json")
        return [
            example_result
            for example_result in (fetch_result_from_file_name(file) for file in logs)
            if example_result
        ]

    def _eval_root_directory(self) -> Path:
        path = self._root_directory / "evals"
        path.mkdir(exist_ok=True)
        return path

    def _eval_directory(self, evaluation_id: str) -> Path:
        path = self._eval_root_directory() / evaluation_id
        path.mkdir(exist_ok=True)
        return path

    def _example_result_path(self, evaluation_id: str, example_id: str) -> Path:
        return (self._eval_directory(evaluation_id) / example_id).with_suffix(".json")

    def _evaluation_run_overview_path(self, evaluation_id: str) -> Path:
        return self._eval_directory(evaluation_id).with_suffix(".json")


class InMemoryEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in memory.

    Preferred for quick testing or notebook use.
    """

    def __init__(self) -> None:
        self._example_evaluations: dict[str, list[ExampleEvaluation[BaseModel]]] = (
            defaultdict(list)
        )
        self._evaluation_run_overviews: dict[str, EvaluationOverview] = dict()

    def store_evaluation_overview(self, overview: EvaluationOverview) -> None:
        self._evaluation_run_overviews[overview.id] = overview

    def evaluation_overview(self, evaluation_id: str) -> Optional[EvaluationOverview]:
        return self._evaluation_run_overviews.get(evaluation_id, None)

    def evaluation_overview_ids(self) -> Sequence[str]:
        return list(self._evaluation_run_overviews.keys())

    def store_example_evaluation(
        self, evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        self._example_evaluations[evaluation.evaluation_id].append(evaluation)

    def example_evaluation(
        self, evaluation_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        return next(
            (
                result
                for result in self.example_evaluations(evaluation_id, evaluation_type)
                if result.example_id == example_id
            ),
            None,
        )

    def example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        return [
            cast(ExampleEvaluation[Evaluation], example_evaluation)
            for example_evaluation in self._example_evaluations[evaluation_id]
        ]


class RecordDataSequence(BaseModel):
    records: Sequence[RecordData]


class ArgillaEvaluationRepository(EvaluationRepository):
    """Evaluation repository used for the :class:`ArgillaEvaluator`.

    Wraps an :class:`Evaluator`.
    Does not support storing evaluations, since the ArgillaEvaluator does not do automated evaluations.

    Args:
        evaluation_repository: repository to wrap.
        argilla_client: client used to connect to Argilla.
        workspace_id: The workspace id to save the datasets in. Has to be created before in Argilla.
        fields: The Argilla fields of the dataset.
        questions: The questions that will be presented to the human evaluators.
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

    def create_evaluation_dataset(self) -> str:
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
        return self._evaluation_repository.evaluation_overview_ids()

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
        self, evaluation_id: str, eval_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        assert eval_type == ArgillaEvaluation
        # Mypy does not derive that the return type is always ExampleEvaluation with ArgillaEvaluation
        return [
            ExampleEvaluation(evaluation_id=evaluation_id, example_id=e.example_id, result=e)  # type: ignore
            for e in self._client.evaluations(evaluation_id)
        ]

    def failed_example_evaluations(
        self, evaluation_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        return self._evaluation_repository.failed_example_evaluations(
            evaluation_id, evaluation_type
        )
