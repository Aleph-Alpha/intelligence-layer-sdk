from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.evaluation import Evaluator, Example, FileDatasetRepository
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    FileAggregationRepository,
)
from intelligence_layer.evaluation.run import main

load_dotenv()


@fixture
def examples() -> Sequence[Example[None, None]]:
    return [Example(input=None, expected_output=None)]


class DummyEvaluation(BaseModel):
    correct: bool


class DummyAggregation(BaseModel):
    correct_rate: float


class DummyTask(Task[None, None]):
    def __init__(self) -> None:
        pass

    def do_run(self, input: None, task_span: TaskSpan) -> None:
        return input


class DummyTaskWithClient(DummyTask):
    def __init__(self, client: AlephAlphaClientProtocol) -> None:
        pass


class DummyEvaluator(Evaluator[None, None, None, DummyEvaluation, DummyAggregation]):
    # mypy expects *args where this method only uses one output
    def do_evaluate(  # type: ignore
        self, input: None, expected_output: None, output: None
    ) -> DummyEvaluation:
        return DummyEvaluation(correct=True)

    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> DummyAggregation:
        list(evaluations)
        return DummyAggregation(correct_rate=1.0)


def test_run_evaluation(
    tmp_path: Path, examples: Sequence[Example[None, None]]
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_repository = FileDatasetRepository(dataset_path)
    dataset_id = dataset_repository.create_dataset(examples)

    aggregation_path = tmp_path / "eval"
    aggregation_repository = FileAggregationRepository(aggregation_path)

    main(
        [
            "",
            "--evaluator",
            "tests.evaluation.test_run.DummyEvaluator",
            "--task",
            "tests.evaluation.test_run.DummyTask",
            "--dataset-repository-path",
            str(dataset_path),
            "--dataset-id",
            dataset_id,
            "--target-dir",
            str(aggregation_path),
            "--description",
            "dummy-evaluator",
        ]
    )
    ids = aggregation_repository.aggregation_ids()
    assert len(ids) == 1
    overview = aggregation_repository.aggregation_overview(ids[0], DummyAggregation)
    assert overview
    assert overview.successful_evaluation_count == 1


def test_run_evaluation_with_task_with_client(
    tmp_path: Path, examples: Sequence[Example[None, None]]
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_repository = FileDatasetRepository(dataset_path)
    dataset_id = dataset_repository.create_dataset(examples)

    eval_path = tmp_path / "eval"

    main(
        [
            "",
            "--evaluator",
            "tests.evaluation.test_run.DummyEvaluator",
            "--task",
            "tests.evaluation.test_run.DummyTaskWithClient",
            "--dataset-repository-path",
            str(dataset_path),
            "--dataset-id",
            dataset_id,
            "--target-dir",
            str(eval_path),
            "--description",
            "dummy-evaluator",
        ]
    )
