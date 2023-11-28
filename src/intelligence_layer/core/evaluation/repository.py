from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.core.evaluation.domain import (
    AggregatedEvaluation,
    Evaluation,
    EvaluationException,
    EvaluationRunOverview,
    ExampleResult,
    TaskSpanTrace,
)
from intelligence_layer.core.evaluation.evaluator import EvaluationRepository
from intelligence_layer.core.tracer import JsonSerializer


class SerializedExampleResult(BaseModel):
    example_id: str
    is_exception: bool
    json_result: str
    trace: TaskSpanTrace

    @classmethod
    def from_example_result(
        cls, result: ExampleResult[Evaluation]
    ) -> "SerializedExampleResult":
        return cls(
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, EvaluationException),
            trace=result.trace,
            example_id=result.example_id,
        )

    def to_example_result(
        self, evaluation_type: type[Evaluation]
    ) -> ExampleResult[Evaluation]:
        if self.is_exception:
            return ExampleResult(
                example_id=self.example_id,
                result=EvaluationException.model_validate_json(self.json_result),
                trace=self.trace,
            )
        else:
            return ExampleResult(
                example_id=self.example_id,
                result=evaluation_type.model_validate_json(self.json_result),
                trace=self.trace,
            )


class FileEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in json-files.

    Args:
        root_directory: The folder where the json-files are stored. The folder (along with its parents)
            will be created if it does not exist yet.
    """

    def __init__(self, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory

    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        def fetch_result_from_file_name(
            path: Path,
        ) -> Optional[ExampleResult[Evaluation]]:
            id = path.with_suffix("").name
            return self.evaluation_example_result(run_id, id, evaluation_type)

        path = self._root_directory / run_id
        logs = path.glob("*.json")
        return [
            example_result
            for example_result in (fetch_result_from_file_name(file) for file in logs)
            if example_result
        ]

    def failed_evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        results = self.evaluation_run_results(run_id, evaluation_type)
        return [r for r in results if isinstance(r.result, EvaluationException)]

    def evaluation_example_result(
        self, run_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleResult[Evaluation]]:
        file_path = (self._root_directory / run_id / example_id).with_suffix(".json")
        if not file_path.exists():
            return None
        content = file_path.read_text()
        serialized_example = SerializedExampleResult.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        run_path = self._root_directory / run_id
        run_path.mkdir(exist_ok=True)
        serialized_result = SerializedExampleResult.from_example_result(result)
        (run_path / result.example_id).with_suffix(".json").write_text(
            serialized_result.model_dump_json(indent=2)
        )

    def evaluation_run_overview(
        self, run_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[EvaluationRunOverview[AggregatedEvaluation]]:
        file_path = (self._root_directory / run_id).with_suffix(".json")
        if not file_path.exists():
            return None
        content = file_path.read_text()
        # Mypy does not accept dynamic types
        return EvaluationRunOverview[aggregation_type].model_validate_json(content)  # type: ignore

    def store_evaluation_run_overview(
        self, overview: EvaluationRunOverview[AggregatedEvaluation]
    ) -> None:
        (self._root_directory / overview.id).with_suffix(".json").write_text(
            overview.model_dump_json(indent=2)
        )

    def run_ids(self) -> Sequence[str]:
        return [
            path.with_suffix("").name for path in self._root_directory.glob("*.json")
        ]


class InMemoryEvaluationRepository(EvaluationRepository):
    _example_results: dict[str, list[str]] = defaultdict(list)

    _run_overviews: dict[str, str] = dict()

    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        result_jsons = self._example_results.get(run_id, [])
        return [
            SerializedExampleResult.model_validate_json(json_str).to_example_result(
                evaluation_type
            )
            for json_str in result_jsons
        ]

    def failed_evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        results = self.evaluation_run_results(run_id, evaluation_type)
        return [r for r in results if isinstance(r.result, EvaluationException)]

    def evaluation_example_result(
        self, run_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> ExampleResult[Evaluation] | None:
        return next(
            (
                result
                for result in self.evaluation_run_results(run_id, evaluation_type)
                if result.example_id == example_id
            ),
            None,
        )

    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        json_result = SerializedExampleResult.from_example_result(result)
        self._example_results[run_id].append(json_result.model_dump_json())

    def store_evaluation_run_overview(
        self, overview: EvaluationRunOverview[AggregatedEvaluation]
    ) -> None:
        self._run_overviews[overview.id] = overview.model_dump_json()

    def evaluation_run_overview(
        self, run_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> EvaluationRunOverview[AggregatedEvaluation] | None:
        loaded_json = self._run_overviews.get(run_id)
        # mypy doesn't accept dynamic types as type parameter
        return (
            EvaluationRunOverview[aggregation_type].model_validate_json(loaded_json)  # type: ignore
            if loaded_json
            else None
        )

    def run_ids(self) -> Sequence[str]:
        return list(self._run_overviews.keys())
