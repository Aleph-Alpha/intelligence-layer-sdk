from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from pydantic import BaseModel

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import (
    Dataset,
    EvaluationRepository,
    Evaluator,
    Example,
    FileEvaluationRepository,
    SequenceDataset,
    Task,
)
from intelligence_layer.core.evaluation.domain import EvaluationOverview
from intelligence_layer.core.evaluation.run import main
from intelligence_layer.core.tracer import TaskSpan

load_dotenv()


def dataset() -> Dataset[None, None]:
    return SequenceDataset(
        name="dummy_dataset", examples=[Example(input=None, expected_output=None)]
    )


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
    def __init__(
        self, task: Task[None, None], repository: EvaluationRepository
    ) -> None:
        super().__init__(task, repository)

    # mypy expects *args where this method only uses one output
    def do_evaluate(  # type: ignore
        self, input: None, expected_output: None, output: None
    ) -> DummyEvaluation:
        return DummyEvaluation(correct=True)

    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> DummyAggregation:
        list(evaluations)
        return DummyAggregation(correct_rate=1.0)

    def evaluation_type(self) -> type[DummyEvaluation]:
        return DummyEvaluation

    def output_type(self) -> type[None]:
        return type(None)


def test_run_evaluation(tmp_path: Path) -> None:
    main(
        [
            "",
            "--evaluator",
            "tests.core.evaluation.test_run.DummyEvaluator",
            "--task",
            "tests.core.evaluation.test_run.DummyTask",
            "--dataset",
            "tests.core.evaluation.test_run.dataset",
            "--target-dir",
            str(tmp_path),
        ]
    )
    repository = FileEvaluationRepository(tmp_path)
    eval_ids = repository.eval_ids()
    assert len(eval_ids) == 1
    overview = repository.evaluation_overview(
        eval_ids[0], EvaluationOverview[DummyAggregation]
    )
    assert overview
    assert overview.successful_count == 1


def test_run_evaluation_with_task_with_client(tmp_path: Path) -> None:
    main(
        [
            "",
            "--evaluator",
            "tests.core.evaluation.test_run.DummyEvaluator",
            "--task",
            "tests.core.evaluation.test_run.DummyTaskWithClient",
            "--dataset",
            "tests.core.evaluation.test_run.dataset",
            "--target-dir",
            str(tmp_path),
        ]
    )
