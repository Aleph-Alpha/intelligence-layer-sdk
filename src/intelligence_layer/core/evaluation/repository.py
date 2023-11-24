from collections import defaultdict
from typing import Sequence
from pydantic import BaseModel
from intelligence_layer.core.evaluation.domain import AggregatedEvaluation, Evaluation, EvaluationException, EvaluationRunOverview, ExampleResult, TaskSpanTrace
from intelligence_layer.core.evaluation.evaluator import EvaluationRepository
from intelligence_layer.core.tracer import JsonSerializer


class InMemoryEvaluationRepository(EvaluationRepository):
    class SerializedExampleResult(BaseModel):
        example_id: str
        is_exception: bool
        json_result: str
        trace: TaskSpanTrace

    _example_results: dict[str, list[str]] = defaultdict(list)

    _run_overviews: dict[str, str] = dict()

    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        def to_example_result(
            serialized_example: InMemoryEvaluationRepository.SerializedExampleResult,
        ) -> ExampleResult[Evaluation]:
            return (
                ExampleResult(
                    example_id=serialized_example.example_id,
                    result=evaluation_type.model_validate_json(
                        serialized_example.json_result
                    ),
                    trace=serialized_example.trace,
                )
                if not serialized_example.is_exception
                else ExampleResult(
                    example_id=serialized_example.example_id,
                    result=EvaluationException.model_validate_json(
                        serialized_example.json_result
                    ),
                    trace=serialized_example.trace,
                )
            )

        result_jsons = self._example_results.get(run_id, [])
        return [
            to_example_result(
                self.SerializedExampleResult.model_validate_json(json_str)
            )
            for json_str in result_jsons
        ]

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
        json_result = self.SerializedExampleResult(
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, EvaluationException),
            trace=result.trace,
            example_id=result.example_id,
        )
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
