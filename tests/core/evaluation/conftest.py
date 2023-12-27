from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    EvaluationOverview,
    Example,
    ExampleEvaluation,
    FailedExampleEvaluation,
    FileEvaluationRepository,
    InMemoryEvaluationRepository,
    RunOverview,
)
from intelligence_layer.core.evaluation.dataset_repository import (
    InMemoryDatasetRepository,
)
from intelligence_layer.core.evaluation.evaluator import DatasetRepository
from intelligence_layer.core.evaluation.runner import Runner
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import Tracer
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
def eval_id() -> str:
    return "eval_id"


@fixture
def failed_example_result(eval_id: str) -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        eval_id=eval_id,
        example_id=0,
        result=FailedExampleEvaluation(error_message="error"),
    )


@fixture
def successful_example_result(eval_id: str) -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        eval_id=eval_id,
        example_id=1,
        result=DummyEvaluation(result="result"),
    )


@fixture
def file_evaluation_repository(tmp_path: Path) -> FileEvaluationRepository:
    return FileEvaluationRepository(tmp_path)


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
def evaluation_run_overview(
    eval_id: str,
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
) -> EvaluationOverview[DummyAggregatedEvaluation]:
    now = datetime.now()
    return EvaluationOverview(
        id=eval_id,
        run_overviews=[
            RunOverview(
                dataset_id="dataset",
                id="run-id",
                start=now,
                end=now,
                failed_example_count=0,
                successful_example_count=0,
                runner_id="dummy-run-id",
            )
        ],
        start=now,
        end=now,
        failed_evaluation_count=3,
        successful_count=5,
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
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> Runner[str, str]:
    return Runner(
        DummyTask(),
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "dummy-runner",
    )
