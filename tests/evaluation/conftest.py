from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import Tracer, utc_now
from intelligence_layer.core.task import Task
from intelligence_layer.evaluation import (
    AggregationOverview,
    DatasetRepository,
    Example,
    ExampleEvaluation,
    FailedExampleEvaluation,
    FileAggregationRepository,
    FileEvaluationRepository,
    FileRunRepository,
    InMemoryDatasetRepository,
    InMemoryRunRepository,
    InstructComparisonArgillaAggregationLogic,
    Runner,
    RunOverview,
)
from intelligence_layer.evaluation.domain import EvaluationOverview
from tests.conftest import DummyStringInput, DummyStringOutput

FAIL_IN_EVAL_INPUT = "fail in eval"
FAIL_IN_TASK_INPUT = "fail in task"


class DummyTask(Task[str, str]):
    def do_run(self, input: str, tracer: Tracer) -> str:
        if input == FAIL_IN_TASK_INPUT:
            raise RuntimeError(input)
        return input


class DummyStringEvaluation(BaseModel):
    same: bool


class DummyEvaluation(BaseModel):
    result: str


class DummyAggregatedEvaluation(BaseModel):
    score: float


class DummyAggregatedEvaluationWithResultList(BaseModel):
    results: Sequence[DummyEvaluation]


@fixture
def evaluation_id() -> str:
    return "evaluation-id-1"


@fixture
def failed_example_result(evaluation_id: str) -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        evaluation_id=evaluation_id,
        example_id="failed-example",
        result=FailedExampleEvaluation(error_message="error"),
    )


@fixture
def successful_example_result(evaluation_id: str) -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        evaluation_id=evaluation_id,
        example_id="successful_example",
        result=DummyEvaluation(result="result"),
    )


@fixture
def file_aggregation_repository(tmp_path: Path) -> FileAggregationRepository:
    return FileAggregationRepository(tmp_path)


@fixture
def file_evaluation_repository(tmp_path: Path) -> FileEvaluationRepository:
    return FileEvaluationRepository(tmp_path)


@fixture
def file_run_repository(tmp_path: Path) -> FileRunRepository:
    return FileRunRepository(tmp_path)


@fixture
def string_dataset_id(
    dummy_string_examples: Iterable[Example[DummyStringInput, DummyStringOutput]],
    in_memory_dataset_repository: DatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(dummy_string_examples)


@fixture
def dummy_aggregated_evaluation() -> DummyAggregatedEvaluation:
    return DummyAggregatedEvaluation(score=0.5)


@fixture
def run_overview() -> RunOverview:
    return RunOverview(
        dataset_id="dataset-id",
        id="run-id-1",
        start=utc_now(),
        end=utc_now(),
        failed_example_count=0,
        successful_example_count=3,
        description="test run overview 1",
    )


@fixture
def evaluation_overview(
    evaluation_id: str, run_overview: RunOverview
) -> EvaluationOverview:
    return EvaluationOverview(
        id=evaluation_id,
        start=utc_now(),
        run_overviews=frozenset([run_overview]),
        description="test evaluation overview 1",
    )


@fixture
def aggregation_overview(
    evaluation_overview: EvaluationOverview,
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
) -> AggregationOverview[DummyAggregatedEvaluation]:
    now = datetime.now()
    return AggregationOverview(
        evaluation_overviews=frozenset([evaluation_overview]),
        id="aggregation-id",
        start=now,
        end=now,
        successful_evaluation_count=5,
        crashed_during_eval_count=3,
        description="dummy-evaluator",
        statistics=dummy_aggregated_evaluation,
    )


@fixture
def dummy_string_example() -> Example[DummyStringInput, DummyStringOutput]:
    return Example(
        input=DummyStringInput.any(), expected_output=DummyStringOutput.any()
    )


@fixture
def dummy_string_examples(
    dummy_string_example: Example[DummyStringInput, DummyStringOutput]
) -> Iterable[Example[DummyStringInput, DummyStringOutput]]:
    return [dummy_string_example]


@fixture
def dummy_runner(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> Runner[str, str]:
    return Runner(
        DummyTask(),
        in_memory_dataset_repository,
        in_memory_run_repository,
        "dummy-runner",
    )


@fixture
def argilla_aggregation_logic() -> InstructComparisonArgillaAggregationLogic:
    return InstructComparisonArgillaAggregationLogic()
