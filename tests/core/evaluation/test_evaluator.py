from typing import Iterable, Literal, TypeAlias

from pydantic import BaseModel
from pytest import fixture, raises

from intelligence_layer.core import (
    Evaluator,
    FailedExampleEvaluation,
    Example,
    InMemoryDatasetRepository,
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
    def do_evaluate(  # type: ignore
        self,
        input: DummyTaskInput,
        expected_output: None,
        output: DummyTaskOutput,
    ) -> DummyEvaluation:
        if output == "fail in eval":
            raise RuntimeError(output)
        return DummyEvaluation(result="pass")

    def aggregate(
        self, evaluations: Iterable[DummyEvaluation]
    ) -> DummyAggregatedEvaluationWithResultList:
        return DummyAggregatedEvaluationWithResultList(results=list(evaluations))


class ComparisonEvaluation(BaseModel):
    is_equal: bool


class ComparisonAggregation(BaseModel):
    equal_ratio: float


class ComparingEvaluator(
    Evaluator[
        DummyTaskInput,
        DummyTaskOutput,
        None,
        ComparisonEvaluation,
        ComparisonAggregation,
    ]
):
    def do_evaluate(
        self, input: DummyTaskInput, expected_output: None, *output: DummyTaskOutput
    ) -> ComparisonEvaluation:
        return ComparisonEvaluation(is_equal=output[1:] == output[:-1])

    def aggregate(
        self, evaluations: Iterable[ComparisonEvaluation]
    ) -> ComparisonAggregation:
        evals = list(evaluations)
        return ComparisonAggregation(
            equal_ratio=evals.count(ComparisonEvaluation(is_equal=True)) / len(evals)
        )


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


@fixture
def dummy_evaluator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> DummyEvaluator:
    return DummyEvaluator(
        DummyTask(),
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
    )


@fixture
def dataset_name(
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    dataset_name = "name"
    in_memory_dataset_repository.create_dataset(
        dataset_name, [(e.input, None) for e in sequence_dataset.examples]
    )
    return dataset_name


@fixture
def comparing_evaluator(
    evaluation_repository: InMemoryEvaluationRepository,
) -> ComparingEvaluator:
    return ComparingEvaluator(DummyTask(), evaluation_repository)


def test_evaluate_dataset_returns_generic_statistics(
    dummy_evaluator: DummyEvaluator, dataset_name: str
) -> None:
    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset_name)

    assert evaluation_run_overview.run_overviews[0].dataset_name == dataset_name
    assert evaluation_run_overview.successful_count == 1
    assert evaluation_run_overview.failed_count == 2


def test_evaluate_dataset_uses_passed_tracer(
    dummy_evaluator: DummyEvaluator, dataset_name: str
) -> None:
    in_memory_tracer = InMemoryTracer()
    dummy_evaluator.evaluate_dataset(dataset_name, in_memory_tracer)

    entries = in_memory_tracer.entries
    assert len(entries) == 3
    assert all([isinstance(e, InMemoryTaskSpan) for e in entries])


def test_evaluate_dataset_saves_overview(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    dataset_name: str,
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
) -> None:
    dataset_name = "datset"
    dataset_repository = InMemoryDatasetRepository()
    dataset_repository.create_dataset(
        dataset_name, [(e.input, None) for e in sequence_dataset.examples]
    )
    dummy_evaluator = DummyEvaluator(
        DummyTask(), in_memory_evaluation_repository, dataset_repository
    )
    overview = dummy_evaluator.evaluate_dataset(dataset_name)

    assert overview == in_memory_evaluation_repository.evaluation_overview(
        overview.id, EvaluationOverview[DummyAggregatedEvaluationWithResultList]
    )


def test_evaluate_dataset_stores_example_evaluations(
    dummy_evaluator: DummyEvaluator,
    dataset_name: str,
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository
    dataset_repository = dummy_evaluator._dataset_repository
    dataset_name = dataset_repository.list_datasets()[0]
    dataset = dataset_repository.dataset(dataset_name, DummyTaskInput, None)
    assert dataset

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(
        dataset_name, NoOpTracer()
    )
    success_result = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, dataset.examples[0].id, DummyEvaluation
    )
    failure_result_task = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, dataset.examples[1].id, DummyEvaluation
    )
    failure_result_eval = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, dataset.examples[2].id, DummyEvaluation
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result_task is None
    assert failure_result_eval and isinstance(
        failure_result_eval.result, FailedExampleEvaluation
    )


def test_evaluate_dataset_stores_example_traces(
    dummy_evaluator: DummyEvaluator,
    dataset_name: str,
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository
    dataset_repository = dummy_evaluator._dataset_repository
    dataset = dataset_repository.dataset(dataset_name, DummyTaskInput, None)
    assert dataset

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset_name)
    success_result = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], dataset.examples[0].id
    )
    failure_result_task = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], dataset.examples[1].id
    )
    failure_result_eval = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], dataset.examples[2].id
    )

    assert success_result
    assert failure_result_task
    assert failure_result_eval
    assert success_result.trace.input == "success"
    assert failure_result_task.trace.input == "fail in task"
    assert failure_result_eval.trace.input == "fail in eval"


def test_evaluate_dataset_stores_aggregated_results(
    dummy_evaluator: DummyEvaluator,
    dataset_name: str,
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository
    dataset_repository = dummy_evaluator._dataset_repository

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(
        dataset_name, NoOpTracer()
    )
    loaded_evaluation_run_overview = evaluation_repository.evaluation_overview(
        evaluation_run_overview.id,
        EvaluationOverview[DummyAggregatedEvaluationWithResultList],
    )

    assert evaluation_run_overview == loaded_evaluation_run_overview


def test_evaluate_can_evaluate_multiple_runs(
    dummy_evaluator: DummyEvaluator,
    comparing_evaluator: ComparingEvaluator,
    sequence_dataset: SequenceDataset[DummyTaskInput, None],
) -> None:
    run_overview1 = dummy_evaluator.run_dataset(sequence_dataset)
    run_overview2 = dummy_evaluator.run_dataset(sequence_dataset)

    partial_overview = comparing_evaluator.evaluate_run(
        sequence_dataset, run_overview1.id, run_overview2.id
    )

    eval_overview = comparing_evaluator.aggregate_evaluation(partial_overview.id)
    assert eval_overview.statistics.equal_ratio == 1


def test_output_type_raises_if_do_run_does_not_have_type_hints() -> None:
    dummy_evaluator = DummyEvaluator(
        DummyTaskWithoutTypeHints(),
        InMemoryEvaluationRepository(),
        InMemoryDatasetRepository(),
    )

    with raises(TypeError):
        dummy_evaluator.output_type()


def test_evaluation_type_raises_if_do_evaluate_does_not_have_type_hints() -> None:
    dummy_evaluator = DummyEvaluatorWithoutTypeHints(
        DummyTask(), InMemoryEvaluationRepository(), InMemoryDatasetRepository()
    )

    with raises(TypeError):
        dummy_evaluator.evaluation_type()
