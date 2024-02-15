from typing import Generic, Iterable, Optional, TypeVar

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import InMemoryTaskSpan, InMemoryTracer, NoOpTracer, Tracer
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.evaluation import (
    BaseEvaluator,
    Evaluation,
    Evaluator,
    Example,
    ExpectedOutput,
    FailedExampleEvaluation,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    MeanAccumulator,
    Runner,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    InMemoryAggregationRepository,
)
from tests.evaluation.conftest import (
    FAIL_IN_EVAL_INPUT,
    FAIL_IN_TASK_INPUT,
    DummyAggregatedEvaluationWithResultList,
    DummyEvaluation,
)


class DummyEvaluator(
    Evaluator[
        str,
        str,
        None,
        DummyEvaluation,
        DummyAggregatedEvaluationWithResultList,
    ]
):
    # mypy expects *args where this method only uses one output
    def do_evaluate(  # type: ignore
        self,
        input: str,
        expected_output: None,
        output: str,
    ) -> DummyEvaluation:
        if output == FAIL_IN_EVAL_INPUT:
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
        str,
        str,
        None,
        ComparisonEvaluation,
        ComparisonAggregation,
    ]
):
    def do_evaluate(
        self, input: str, expected_output: None, *output: str
    ) -> ComparisonEvaluation:
        return ComparisonEvaluation(is_equal=output[1:] == output[:-1])

    def aggregate(
        self, evaluations: Iterable[ComparisonEvaluation]
    ) -> ComparisonAggregation:
        acc = MeanAccumulator()
        for evaluation in evaluations:
            acc.add(1.0 if evaluation.is_equal else 0.0)
        return ComparisonAggregation(equal_ratio=acc.extract())


class DummyEvaluatorWithoutTypeHints(DummyEvaluator):
    # type hint for return value missing on purpose for testing
    def do_evaluate(  # type: ignore
        self, input: str, output: str, expected_output: None
    ):
        return super().do_evaluate(input, expected_output, output)


class DummyTaskWithoutTypeHints(Task[str, str]):
    # type hint for return value missing on purpose for testing
    def do_run(self, input: str, tracer: Tracer):  # type: ignore
        return input


@fixture
def sequence_examples() -> Iterable[Example[str, None]]:
    return [
        Example(input="success", expected_output=None),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None),
    ]


@fixture
def sequence_good_examples() -> Iterable[Example[str, None]]:
    return [
        Example(input="success", expected_output=None),
        Example(input="success", expected_output=None),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None),
    ]


@fixture
def dummy_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
) -> DummyEvaluator:
    return DummyEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "dummy-evaluator",
    )


@fixture
def dataset_id(
    sequence_examples: Iterable[Example[str, None]],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(sequence_examples)


@fixture
def good_dataset_id(
    sequence_good_examples: Iterable[Example[str, None]],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(sequence_good_examples)


@fixture
def comparing_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
) -> ComparingEvaluator:
    return ComparingEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "comparing-evaluator",
    )


def test_eval_and_aggregate_runs_returns_generic_statistics(
    dummy_evaluator: DummyEvaluator, dummy_runner: Runner[str, str], dataset_id: str
) -> None:
    run_overview = dummy_runner.run_dataset(dataset_id)
    aggregation_overview = dummy_evaluator.eval_and_aggregate_runs(run_overview.id)

    assert next(iter(aggregation_overview.run_overviews())).dataset_id == dataset_id
    assert aggregation_overview.successful_evaluation_count == 1
    assert aggregation_overview.failed_evaluation_count == 2


def test_eval_and_aggregate_runs_uses_passed_tracer(
    dummy_evaluator: DummyEvaluator, dataset_id: str, dummy_runner: Runner[str, str]
) -> None:
    in_memory_tracer = InMemoryTracer()
    run_overview = dummy_runner.run_dataset(dataset_id, in_memory_tracer)
    dummy_evaluator.eval_and_aggregate_runs(run_overview.id)

    entries = in_memory_tracer.entries
    assert len(entries) == 3
    assert all([isinstance(e, InMemoryTaskSpan) for e in entries])


def test_eval_and_aggregate_runs_stores_example_evaluations(
    dummy_evaluator: DummyEvaluator, dataset_id: str, dummy_runner: Runner[str, str]
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository
    dataset_repository = dummy_evaluator._dataset_repository
    dataset_id = list(dataset_repository.list_datasets())[0]
    dataset: Optional[Iterable[Example[str, None]]] = dataset_repository.examples_by_id(
        dataset_id, str, type(None)
    )
    assert dataset is not None

    run_overview = dummy_runner.run_dataset(dataset_id, NoOpTracer())
    aggregation_overview = dummy_evaluator.eval_and_aggregate_runs(run_overview.id)
    examples = list(dataset)
    eval_overview = next(iter(aggregation_overview.evaluation_overviews))
    success_result = evaluation_repository.example_evaluation(
        eval_overview.id,
        examples[0].id,
        DummyEvaluation,
    )
    failure_result_task = evaluation_repository.example_evaluation(
        eval_overview.id,
        examples[1].id,
        DummyEvaluation,
    )
    failure_result_eval = evaluation_repository.example_evaluation(
        eval_overview.id,
        examples[2].id,
        DummyEvaluation,
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result_task is None
    assert failure_result_eval and isinstance(
        failure_result_eval.result, FailedExampleEvaluation
    )


def test_eval_and_aggregate_runs_stores_example_traces(
    dummy_evaluator: DummyEvaluator,
    dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    run_repository = dummy_evaluator._run_repository
    dataset_repository = dummy_evaluator._dataset_repository
    dataset: Optional[Iterable[Example[str, None]]] = dataset_repository.examples_by_id(
        dataset_id, str, type(None)
    )
    assert dataset is not None

    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_run_overview = dummy_evaluator.eval_and_aggregate_runs(run_overview.id)
    examples = list(dataset)
    success_result = run_repository.example_trace(
        evaluation_run_overview.run_ids[0], examples[0].id
    )
    failure_result_task = run_repository.example_trace(
        evaluation_run_overview.run_ids[0], examples[1].id
    )
    failure_result_eval = run_repository.example_trace(
        evaluation_run_overview.run_ids[0], examples[2].id
    )

    assert success_result
    assert failure_result_task
    assert failure_result_eval
    assert success_result.trace.input == "success"
    assert failure_result_task.trace.input == FAIL_IN_TASK_INPUT
    assert failure_result_eval.trace.input == FAIL_IN_EVAL_INPUT


def test_eval_and_aggregate_runs_stores_aggregated_results(
    dummy_evaluator: DummyEvaluator,
    dummy_runner: Runner[str, str],
    dataset_id: str,
) -> None:
    aggregation_repository = dummy_evaluator._aggregation_repository

    run_overview = dummy_runner.run_dataset(dataset_id)
    aggregation_overview = dummy_evaluator.eval_and_aggregate_runs(run_overview.id)
    loaded_evaluation_run_overview = aggregation_repository.aggregation_overview(
        aggregation_overview.id, DummyAggregatedEvaluationWithResultList
    )

    assert aggregation_overview == loaded_evaluation_run_overview


def test_evaluate_can_evaluate_multiple_runs(
    comparing_evaluator: ComparingEvaluator,
    string_dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    run_overview1 = dummy_runner.run_dataset(string_dataset_id)
    run_overview2 = dummy_runner.run_dataset(string_dataset_id)

    partial_overview = comparing_evaluator.evaluate_runs(
        run_overview1.id, run_overview2.id
    )

    eval_overview = comparing_evaluator.aggregate_evaluation(partial_overview.id)
    assert eval_overview.statistics.equal_ratio == 1


def test_aggregate_evaluation_can_aggregate_multiple_evals(
    comparing_evaluator: ComparingEvaluator,
    string_dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    run_overview_1 = dummy_runner.run_dataset(string_dataset_id)
    run_overview_2 = dummy_runner.run_dataset(string_dataset_id)

    partial_overview_1 = comparing_evaluator.evaluate_runs(run_overview_1.id)
    partial_overview_2 = comparing_evaluator.evaluate_runs(
        run_overview_1.id, run_overview_2.id
    )

    aggregation_overview = comparing_evaluator.aggregate_evaluation(
        partial_overview_1.id, partial_overview_1.id, partial_overview_2.id
    )

    assert len(list(aggregation_overview.run_overviews())) == 2
    assert aggregation_overview.statistics.equal_ratio == 1


def test_base_evaluator_type_magic_works(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
) -> None:
    class EvaluationType(BaseModel):
        pass

    class AggregatedEvaluationType(BaseModel):
        pass

    types = {
        "Input": str,
        "Output": str,
        "ExpectedOutput": type(None),
        "Evaluation": EvaluationType,
        "AggregatedEvaluation": AggregatedEvaluationType,
    }

    class ChildEvaluator(
        BaseEvaluator[Input, Output, None, Evaluation, AggregatedEvaluationType]
    ):
        def evaluate(
            self,
            example: Example[Input, ExpectedOutput],
            eval_id: str,
            *outputs: SuccessfulExampleOutput[Output],
        ) -> None:
            pass

        def aggregate(
            self, evaluations: Iterable[Evaluation]
        ) -> AggregatedEvaluationType:
            return AggregatedEvaluationType()

    A = TypeVar("A", bound=BaseModel)

    class GrandChildEvaluator(ChildEvaluator[Input, str, A]):
        pass

    class Mailman(Generic[A]):
        pass

    class GreatGrandChildEvaluator(
        Mailman[EvaluationType], GrandChildEvaluator[str, EvaluationType]
    ):
        pass

    timmy = GreatGrandChildEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "dummy",
    )
    who_is_timmy = timmy._get_types()

    assert who_is_timmy == types


def test_eval_and_aggregate_runs_only_runs_n_examples(
    dummy_evaluator: DummyEvaluator,
    dummy_runner: Runner[str, str],
    good_dataset_id: str,
) -> None:
    run_overview = dummy_runner.run_dataset(good_dataset_id)
    evaluation_overview = dummy_evaluator.eval_and_aggregate_runs(run_overview.id)
    partial_evaluation_overview = dummy_evaluator.evaluate_runs(
        run_overview.id, num_examples=2
    )
    evaluation_overview_n = dummy_evaluator.aggregate_evaluation(
        partial_evaluation_overview.id
    )

    assert (
        evaluation_overview.successful_evaluation_count
        + evaluation_overview.crashed_during_eval_count
        == 3
    )
    assert (
        evaluation_overview_n.successful_evaluation_count
        + evaluation_overview_n.crashed_during_eval_count
        == 2
    )
