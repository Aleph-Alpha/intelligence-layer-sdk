from typing import Iterable, Literal, TypeAlias

from pytest import fixture, raises

from intelligence_layer.core import (
    Evaluator,
    Example,
    FailedExampleEvaluation,
    InMemoryEvaluationRepository,
    InMemoryTaskSpan,
    InMemoryTracer,
    NoOpTracer,
    SequenceDataset,
    Tracer,
)
from intelligence_layer.core.evaluation.domain import EvaluationOverview
from intelligence_layer.core.task import Task
from tests.core.evaluation.conftest import (
    DummyAggregatedEvaluationWithResultList,
    DummyEvaluation,
)

DummyTaskInput: TypeAlias = Literal["success", "fail in task", "fail in eval"]
DummyTaskOutput: TypeAlias = DummyTaskInput


@fixture
def sequence_dataset() -> SequenceDataset[DummyTaskInput, None]:
    examples = [
        Example(input="success", expected_output=None),
        Example(input="fail in task", expected_output=None),
        Example(input="fail in eval", expected_output=None),
    ]
    return SequenceDataset(
        name="test",
        examples=examples,  # type: ignore
    )


class DummyEvaluator(
    Evaluator[
        DummyTaskInput,
        DummyTaskOutput,
        None,
        DummyEvaluation,
        DummyAggregatedEvaluationWithResultList,
    ]
):
    # mypy expects *args where this method only uses one output
    def do_evaluate( # type: ignore
        self, input: DummyTaskInput, expected_output: None, output: DummyTaskOutput,
    ) -> DummyEvaluation:
        if output == "fail in eval":
            raise RuntimeError(output)
        return DummyEvaluation(result="pass")

    def aggregate(
        self, evaluations: Iterable[DummyEvaluation]
    ) -> DummyAggregatedEvaluationWithResultList:
        return DummyAggregatedEvaluationWithResultList(results=list(evaluations))


class DummyEvaluatorWithoutTypeHints(DummyEvaluator):
    # type hint for return value missing on purpose for testing
    def do_evaluate(  # type: ignore
        self, input: DummyTaskInput, output: DummyTaskOutput, expected_output: None
    ):
        return super().do_evaluate(input, expected_output, output)


class DummyTask(Task[DummyTaskInput, DummyTaskOutput]):
    def do_run(self, input: DummyTaskInput, tracer: Tracer) -> DummyTaskOutput:
        if input == "fail in task":
            raise RuntimeError(input)
        return input


class DummyTaskWithoutTypeHints(Task[DummyTaskInput, DummyTaskOutput]):
    # type hint for return value missing on purpose for testing
    def do_run(self, input: DummyTaskInput, tracer: Tracer):  # type: ignore
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
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
) -> None:
    evaluation_run_overview = dummy_evaluator.evaluate_dataset(sequence_dataset)

    assert evaluation_run_overview.run_overviews[0].dataset_name == sequence_dataset.name
    assert evaluation_run_overview.successful_count == 1
    assert evaluation_run_overview.failed_count == 2


def test_evaluate_dataset_uses_passed_tracer(
    dummy_evaluator: DummyEvaluator,
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
) -> None:
    in_memory_tracer = InMemoryTracer()
    dummy_evaluator.evaluate_dataset(sequence_dataset, in_memory_tracer)

    entries = in_memory_tracer.entries
    assert len(entries) == 3
    assert all([isinstance(e, InMemoryTaskSpan) for e in entries])


def test_evaluate_dataset_saves_overview(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
) -> None:
    dummy_evaluator = DummyEvaluator(DummyTask(), in_memory_evaluation_repository)
    overview = dummy_evaluator.evaluate_dataset(sequence_dataset)

    assert overview == in_memory_evaluation_repository.evaluation_overview(
        overview.id, EvaluationOverview[DummyAggregatedEvaluationWithResultList]
    )


def test_evaluate_dataset_stores_example_evaluations(
    dummy_evaluator: DummyEvaluator,
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
) -> None:
    evaluation_repository = dummy_evaluator._repository

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(
        sequence_dataset, NoOpTracer()
    )
    success_result = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, sequence_dataset.examples[0].id, DummyEvaluation
    )
    failure_result_task = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, sequence_dataset.examples[1].id, DummyEvaluation
    )
    failure_result_eval = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, sequence_dataset.examples[2].id, DummyEvaluation
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result_task is None
    assert failure_result_eval and isinstance(
        failure_result_eval.result, FailedExampleEvaluation
    )


def test_evaluate_dataset_stores_example_traces(
    dummy_evaluator: DummyEvaluator,
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
) -> None:
    evaluation_repository = dummy_evaluator._repository

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(sequence_dataset)
    success_result = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], sequence_dataset.examples[0].id
    )
    failure_result_task = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], sequence_dataset.examples[1].id
    )
    failure_result_eval = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], sequence_dataset.examples[2].id
    )

    assert success_result
    assert failure_result_task
    assert failure_result_eval
    assert success_result.trace.input == "success"
    assert failure_result_task.trace.input == "fail in task"
    assert failure_result_eval.trace.input == "fail in eval"


def test_evaluate_dataset_stores_aggregated_results(
    dummy_evaluator: DummyEvaluator,
) -> None:
    evaluation_repository = dummy_evaluator._repository

    dataset: SequenceDataset[DummyTaskInput, None] = SequenceDataset(
        name="test", examples=[]
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    loaded_evaluation_run_overview = evaluation_repository.evaluation_overview(
        evaluation_run_overview.id,
        EvaluationOverview[DummyAggregatedEvaluationWithResultList],
    )

    assert evaluation_run_overview == loaded_evaluation_run_overview


def test_output_type_raises_if_do_run_does_not_have_type_hints(
    evaluation_repository: InMemoryEvaluationRepository,
) -> None:
    dummy_evaluator = DummyEvaluator(DummyTaskWithoutTypeHints(), evaluation_repository)

    with raises(TypeError):
        dummy_evaluator.output_type()


def test_evaluation_type_raises_if_do_evaluate_does_not_have_type_hints(
    evaluation_repository: InMemoryEvaluationRepository,
) -> None:
    dummy_evaluator = DummyEvaluatorWithoutTypeHints(DummyTask(), evaluation_repository)

    with raises(TypeError):
        dummy_evaluator.evaluation_type()
