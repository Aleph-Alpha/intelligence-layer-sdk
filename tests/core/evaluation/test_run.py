from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from intelligence_layer.core import (
    Dataset,
    EvaluationRepository,
    Evaluator,
    Example,
    FileEvaluationRepository,
    SequenceDataset,
    Task,
)
from intelligence_layer.core.evaluation.run import main
from intelligence_layer.core.tracer import TaskSpan


def dataset() -> Dataset[None, None]:
    return SequenceDataset(
        name="dummy_dataset", examples=[Example(input=None, expected_output=None)]
    )


class DummyEvaluation(BaseModel):
    correct: bool


class DummyAggregation(BaseModel):
    correct_rate: float


class DummyTask(Task[None, None]):
    def do_run(self, input: None, task_span: TaskSpan) -> None:
        return input


class DummyEvaluator(Evaluator[None, None, None, DummyEvaluation, DummyAggregation]):
    def __init__(
        self, task: Task[None, None], repository: EvaluationRepository
    ) -> None:
        super().__init__(task, repository)

    def do_evaluate(
        self, input: None, output: None, expected_output: None
    ) -> DummyEvaluation:
        return DummyEvaluation(correct=True)

    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> DummyAggregation:
        list(evaluations)
        return DummyAggregation(correct_rate=1.0)


def test_run_evaluation(tmp_path: Path) -> None:
    main(
        [
            "tests.core.evaluation.test_run.DummyEvaluator",
            "tests.core.evaluation.test_run.DummyTask",
            "tests.core.evaluation.test_run.dataset",
            str(tmp_path),
        ]
    )
    repository = FileEvaluationRepository(tmp_path)
    run_ids = repository.run_ids()
    assert len(run_ids) == 1
    overview = repository.evaluation_run_overview(run_ids[0], DummyAggregation)
    assert overview
    assert overview.successful_evaluation_count == 1
