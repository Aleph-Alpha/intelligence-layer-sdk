from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    EvaluationException,
    Evaluator,
    Example,
    InMemoryEvaluationRepository,
    InMemoryTaskSpan,
    InMemoryTracer,
    NoOpTracer,
    SequenceDataset,
    Tracer,
)
from intelligence_layer.core.task import Task
from tests.core.test_tracer import parse_log

DummyTaskInput = Literal["success", "fail in task", "fail in eval"]
DummyTaskOutput = DummyTaskInput


class DummyEvaluation(BaseModel):
    result: str


class AggregatedDummyEvaluation(BaseModel):
    results: Sequence[DummyEvaluation]


@fixture
def sequence_dataset() -> SequenceDataset[DummyTaskInput, None]:
    examples: Sequence[Example[DummyTaskInput, None]] = [
        Example(input="success", expected_output=None),
    ]
    return SequenceDataset(
        name="test",
        examples=examples,
    )


class DummyEvaluator(
    Evaluator[
        DummyTaskInput,
        DummyTaskOutput,
        None,
        DummyEvaluation,
        AggregatedDummyEvaluation,
    ]
):
    def do_evaluate(
        self, input: DummyTaskInput, output: DummyTaskOutput, expected_output: None
    ) -> DummyEvaluation:
        if output == "fail in eval":
            raise RuntimeError(output)
        return DummyEvaluation(result="pass")

    def aggregate(
        self, evaluations: Iterable[DummyEvaluation]
    ) -> AggregatedDummyEvaluation:
        return AggregatedDummyEvaluation(results=list(evaluations))


class DummyTask(Task[DummyTaskInput, DummyTaskOutput]):
    def do_run(self, input: DummyTaskInput, tracer: Tracer) -> DummyTaskOutput:
        if input == "fail in task":
            raise RuntimeError(input)
        return input


@fixture
def evaluation_repository() -> InMemoryEvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def dummy_evaluator(
    evaluation_repository: InMemoryEvaluationRepository,
) -> DummyEvaluator:
    return DummyEvaluator(DummyTask(), evaluation_repository)


def test_evaluate_dataset_returns_generic_statistics(
    dummy_evaluator: DummyEvaluator,
) -> None:
    test_start = datetime.utcnow()
    examples: Sequence[Example[DummyTaskInput, None]] = [
        Example(input="success", expected_output=None),
        Example(input="fail in task", expected_output=None),
        Example(input="fail in eval", expected_output=None),
    ]

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=examples,
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset)

    assert evaluation_run_overview.dataset_name == dataset.name
    assert test_start <= evaluation_run_overview.start <= evaluation_run_overview.end
    assert evaluation_run_overview.failed_evaluation_count == 2
    assert evaluation_run_overview.successful_evaluation_count == 1


def test_evaluate_dataset_uses_passed_tracer(
    dummy_evaluator: DummyEvaluator,
) -> None:
    examples: Sequence[Example[DummyTaskInput, None]] = [
        Example(input="success", expected_output=None),
        Example(input="fail in task", expected_output=None),
        Example(input="fail in eval", expected_output=None),
    ]

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=examples,
    )
    in_memory_tracer = InMemoryTracer()
    dummy_evaluator.evaluate_dataset(dataset, in_memory_tracer)

    entries = in_memory_tracer.entries
    assert len(entries) == 3
    assert all([isinstance(e, InMemoryTaskSpan) for e in entries])


def test_evaluate_dataset_saves_result_to_file(
    evaluation_repository: InMemoryEvaluationRepository,
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
    tmp_path: Path,
) -> None:
    expected = InMemoryTracer()
    dummy_evaluator = DummyEvaluator(DummyTask(), evaluation_repository, tmp_path)
    overview = dummy_evaluator.evaluate_dataset(sequence_dataset, expected)

    found = parse_log((tmp_path / overview.id).with_suffix(".jsonl"))

    assert found == expected


def test_evaluate_dataset_stores_example_results(
    dummy_evaluator: DummyEvaluator,
) -> None:
    evaluation_repository = dummy_evaluator._repository
    examples: Sequence[Example[DummyTaskInput, None]] = [
        Example(input="success", expected_output=None),
        Example(input="fail in task", expected_output=None),
        Example(input="fail in eval", expected_output=None),
    ]

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=examples,
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    success_result = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[0].id, DummyEvaluation
    )
    failure_result_task = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[1].id, DummyEvaluation
    )
    failure_result_eval = evaluation_repository.evaluation_example_result(
        evaluation_run_overview.id, examples[2].id, DummyEvaluation
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result_task and isinstance(
        failure_result_task.result, EvaluationException
    )
    assert failure_result_eval and isinstance(
        failure_result_eval.result, EvaluationException
    )


def test_evaluate_dataset_stores_example_traces(
    dummy_evaluator: DummyEvaluator,
) -> None:
    evaluation_repository = dummy_evaluator._repository
    examples: Sequence[Example[DummyTaskInput, None]] = [
        Example(input="success", expected_output=None),
        Example(input="fail in task", expected_output=None),
        Example(input="fail in eval", expected_output=None),
    ]

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=examples,
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset)
    success_result = evaluation_repository.evaluation_example_trace(
        evaluation_run_overview.id, examples[0].id
    )
    failure_result_task = evaluation_repository.evaluation_example_trace(
        evaluation_run_overview.id, examples[1].id
    )
    failure_result_eval = evaluation_repository.evaluation_example_trace(
        evaluation_run_overview.id, examples[2].id
    )

    assert success_result.trace.input == "success"
    assert failure_result_task.trace.input == "fail in task"
    assert failure_result_eval.trace.input == "fail in eval"


def test_evaluate_dataset_stores_aggregated_results(
    dummy_evaluator: DummyEvaluator,
) -> None:
    evaluation_repository = dummy_evaluator._repository

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test",
        examples=[],
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    loaded_evaluation_run_overview = evaluation_repository.evaluation_run_overview(
        evaluation_run_overview.id, AggregatedDummyEvaluation
    )

    assert evaluation_run_overview == loaded_evaluation_run_overview
