from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, TypeVar, cast

from pydantic import BaseModel, ValidationError

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
)
from intelligence_layer.core import Output
from intelligence_layer.core.tracer import FileTracer, JsonSerializer, Tracer
from intelligence_layer.evaluation.data_storage.utils import read_utf8, write_utf8
from intelligence_layer.evaluation.domain import (
    Evaluation,
    ExampleEvaluation,
    ExampleOutput,
    FailedExampleEvaluation,
    IndividualEvaluationOverview,
    RunOverview,
)

EvaluationOverviewType = TypeVar(
    "EvaluationOverviewType", bound=IndividualEvaluationOverview
)


class SerializedExampleEvaluation(BaseModel):
    """A json-serialized evaluation of a single example in a dataset.

    Attributes:
        eval_id: Identifier of the run the evaluated example belongs to.
        example_id: Unique identifier of the example this evaluation was created for.
        is_exception: qill be `True` if an exception occurred during evaluation.
        json_result: The actrual serialized evaluation result.
    """

    eval_id: str
    example_id: str
    is_exception: bool
    json_result: str

    @classmethod
    def from_example_result(
        cls, result: ExampleEvaluation[Evaluation]
    ) -> "SerializedExampleEvaluation":
        return cls(
            eval_id=result.eval_id,
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, FailedExampleEvaluation),
            example_id=result.example_id,
        )

    def to_example_result(
        self, evaluation_type: type[Evaluation]
    ) -> ExampleEvaluation[Evaluation]:
        if self.is_exception:
            return ExampleEvaluation(
                eval_id=self.eval_id,
                example_id=self.example_id,
                result=FailedExampleEvaluation.model_validate_json(self.json_result),
            )
        else:
            return ExampleEvaluation(
                eval_id=self.eval_id,
                example_id=self.example_id,
                result=evaluation_type.model_validate_json(self.json_result),
            )


class EvaluationRepository(ABC):
    """Base evaluation repository interface.

    Provides methods to store and load evaluation results for individual examples
    of a run and the aggregated evaluation of said run.
    """

    @abstractmethod
    def eval_ids(self) -> Sequence[str]:
        """Returns the ids of all stored evaluation runs.

        Having the id of an evaluation run, its overview can be retrieved with
        :meth:`EvaluationRepository.evaluation_run_overview`.

        Returns:
            The ids of all stored evaluation runs.
        """
        ...

    @abstractmethod
    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        """Returns an :class:`ExampleEvaluation` of a given run by its id.

        Args:
            eval_id: Identifier of the run to obtain the results for.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
            evaluation_type: Type of evaluations that the `Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            :class:`ExampleEvaluation` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_example_evaluation(self, result: ExampleEvaluation[Evaluation]) -> None:
        """Stores an :class:`ExampleEvaluation` for a run in the repository.

        Args:
            eval_id: Identifier of the eval run.
            result: The result to be persisted.
        """
        ...

    @abstractmethod
    def example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all :class:`ExampleResult` instances of a given run

        Args:
            eval_id: Identifier of the eval run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            All :class:`ExampleResult` of the run. Will return an empty list if there's none.
        """
        ...

    @abstractmethod
    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        """Returns all failed :class:`ExampleResult` instances of a given run

        Args:
            eval_id: Identifier of the eval run to obtain the results for.
            evaluation_type: Type of evaluations that the :class:`Evaluator` returned
                in :func:`Evaluator.do_evaluate`

        Returns:
            All failed :class:`ExampleResult` of the run. Will return an empty list if there's none.
        """
        ...

    @abstractmethod
    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        """Returns an :class:`EvaluationOverview` of a given run by its id.

        Args:
            eval_id: Identifier of the eval run to obtain the overview for.
            aggregation_type: Type of aggregations that the :class:`Evaluator` returned
                in :func:`Evaluator.aggregate`

        Returns:
            :class:`EvaluationOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_evaluation_overview(self, overview: IndividualEvaluationOverview) -> None:
        """Stores an :class:`EvaluationRunOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...


class FileEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in json-files.

    Args:
        root_directory: The folder where the json-files are stored. The folder (along with its parents)
            will be created if it does not exist yet.
    """

    def __init__(self, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory

    def _run_root_directory(self) -> Path:
        path = self._root_directory / "runs"
        path.mkdir(exist_ok=True)
        return path

    def _run_directory(self, run_id: str) -> Path:
        path = self._run_root_directory() / run_id
        path.mkdir(exist_ok=True)
        return path

    def _output_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "output"
        path.mkdir(exist_ok=True)
        return path

    def _trace_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "trace"
        path.mkdir(exist_ok=True)
        return path

    def _eval_root_directory(self) -> Path:
        path = self._root_directory / "evals"
        path.mkdir(exist_ok=True)
        return path

    def _eval_directory(self, eval_id: str) -> Path:
        path = self._eval_root_directory() / eval_id
        path.mkdir(exist_ok=True)
        return path

    def _example_output_path(self, run_id: str, example_id: str) -> Path:
        return (self._output_directory(run_id) / example_id).with_suffix(".json")

    def _example_trace_path(self, run_id: str, example_id: str) -> Path:
        return (self._trace_directory(run_id) / example_id).with_suffix(".jsonl")

    def _example_result_path(self, eval_id: str, example_id: str) -> Path:
        return (self._eval_directory(eval_id) / example_id).with_suffix(".json")

    def _evaluation_run_overview_path(self, eval_id: str) -> Path:
        return self._eval_directory(eval_id).with_suffix(".json")

    def _run_overview_path(self, run_id: str) -> Path:
        return self._run_directory(run_id).with_suffix(".json")

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        serialized_result = JsonSerializer(root=example_output)
        write_utf8(
            self._example_output_path(example_output.run_id, example_output.example_id),
            serialized_result.model_dump_json(indent=2),
        )

    def example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        def fetch_result_from_file_name(
            path: Path,
        ) -> Optional[ExampleEvaluation[Evaluation]]:
            id = path.with_suffix("").name
            return self.example_evaluation(eval_id, id, evaluation_type)

        path = self._eval_directory(eval_id)
        logs = path.glob("*.json")
        return [
            example_result
            for example_result in (fetch_result_from_file_name(file) for file in logs)
            if example_result
        ]

    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        results = self.example_evaluations(eval_id, evaluation_type)
        return [r for r in results if isinstance(r.result, FailedExampleEvaluation)]

    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        file_path = self._example_result_path(eval_id, example_id)
        if not file_path.exists():
            return None
        content = read_utf8(file_path)
        serialized_example = SerializedExampleEvaluation.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        file_path = self._example_trace_path(run_id, example_id)
        return FileTracer(file_path)

    def store_example_evaluation(self, result: ExampleEvaluation[Evaluation]) -> None:
        serialized_result = SerializedExampleEvaluation.from_example_result(result)
        write_utf8(
            self._example_result_path(result.eval_id, result.example_id),
            serialized_result.model_dump_json(indent=2),
        )

    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        file_path = self._evaluation_run_overview_path(eval_id)
        if not file_path.exists():
            return None
        content = read_utf8(file_path)
        return overview_type.model_validate_json(content)

    def store_evaluation_overview(self, overview: IndividualEvaluationOverview) -> None:
        write_utf8(
            self._evaluation_run_overview_path(overview.id),
            overview.model_dump_json(indent=2),
        )

    def store_run_overview(self, overview: RunOverview) -> None:
        write_utf8(
            self._run_overview_path(overview.id), overview.model_dump_json(indent=2)
        )

    def eval_ids(
        self, overview_type: type[EvaluationOverviewType] | None = None
    ) -> Sequence[str]:
        def evaluation_overview(
            eval_id: str, overview_type: type[EvaluationOverviewType]
        ) -> EvaluationOverviewType | None:
            try:
                return self.evaluation_overview(eval_id, overview_type)
            except ValidationError:
                return None

        if overview_type is None:
            return [path.stem for path in self._eval_root_directory().glob("*.json")]
        overviews = (
            evaluation_overview(path.stem, overview_type)
            for path in self._eval_root_directory().glob("*.json")
        )
        return [overview.id for overview in overviews if overview is not None]


class InMemoryEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in memory.

    Preferred for quick testing or notebook use.
    """

    def __init__(self) -> None:
        self._example_evaluations: dict[
            str, list[ExampleEvaluation[BaseModel]]
        ] = defaultdict(list)
        self._evaluation_run_overviews: dict[str, IndividualEvaluationOverview] = dict()

    def eval_ids(self) -> Sequence[str]:
        return list(self._evaluation_run_overviews.keys())

    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> ExampleEvaluation[Evaluation] | None:
        return next(
            (
                result
                for result in self.example_evaluations(eval_id, evaluation_type)
                if result.example_id == example_id
            ),
            None,
        )

    def store_example_evaluation(
        self, evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        self._example_evaluations[evaluation.eval_id].append(evaluation)

    def example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        return [
            cast(ExampleEvaluation[Evaluation], example_evaluation)
            for example_evaluation in self._example_evaluations[eval_id]
        ]

    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        results = self.example_evaluations(eval_id, evaluation_type)
        return [r for r in results if isinstance(r.result, FailedExampleEvaluation)]

    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        return cast(EvaluationOverviewType, self._evaluation_run_overviews[eval_id])

    def store_evaluation_overview(self, overview: IndividualEvaluationOverview) -> None:
        self._evaluation_run_overviews[overview.id] = overview


class ArgillaEvaluationRepository(EvaluationRepository):
    """Evaluation repository used for the :class:`ArgillaEvaluator`.

    Wraps an :class:`Evaluator`.
    Does not support storing evaluations, since the ArgillaEvaluator does not do automated evaluations.

    Args:
        evaluation_repository: repository to wrap.
        argilla_client: client used to connect to Argilla.
    """

    def __init__(
        self, evaluation_repository: EvaluationRepository, argilla_client: ArgillaClient
    ) -> None:
        super().__init__()
        self._evaluation_repository = evaluation_repository
        self._client = argilla_client

    def eval_ids(self) -> Sequence[str]:
        return self._evaluation_repository.eval_ids()

    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        return self._evaluation_repository.example_evaluation(
            eval_id, example_id, evaluation_type
        )

    def store_example_evaluation(self, _: ExampleEvaluation[Evaluation]) -> None:
        raise TypeError(
            "ArgillaEvaluationRepository does not support storing evaluations."
        )

    def example_evaluations(
        self, eval_id: str, eval_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        assert eval_type == ArgillaEvaluation
        # Mypy does not derive that the return type is always ExampleEvaluation with ArgillaEvaluation
        return [
            ExampleEvaluation(eval_id=eval_id, example_id=e.example_id, result=e)  # type: ignore
            for e in self._client.evaluations(eval_id)
        ]

    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        return self._evaluation_repository.failed_example_evaluations(
            eval_id, evaluation_type
        )

    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        return self._evaluation_repository.evaluation_overview(eval_id, overview_type)

    def store_evaluation_overview(self, overview: IndividualEvaluationOverview) -> None:
        return self._evaluation_repository.store_evaluation_overview(overview)
