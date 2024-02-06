from typing import Generic, Iterable, Optional, TypeVar

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    Evaluation,
    EvaluationOverview,
    Evaluator,
    Example,
    ExpectedOutput,
    FailedExampleEvaluation,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryTaskSpan,
    InMemoryTracer,
    NoOpTracer,
    SuccessfulExampleOutput,
    Tracer,
)
from intelligence_layer.core.evaluation.accumulator import MeanAccumulator
from intelligence_layer.core.evaluation.evaluator import BaseEvaluator
from intelligence_layer.core.evaluation.runner import Runner
from intelligence_layer.core.task import Input, Output, Task
from tests.core.evaluation.conftest import (
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
def dummy_evaluator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> DummyEvaluator:
    return DummyEvaluator(
        in_memory_evaluation_repository, in_memory_dataset_repository, "dummy-evaluator"
    )


@fixture
def dataset_id(
    sequence_examples: Iterable[Example[str, None]],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(sequence_examples)


@fixture
def comparing_evaluator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> ComparingEvaluator:
    return ComparingEvaluator(
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "comparing-evaluator",
    )


def test_evaluate_dataset_returns_generic_statistics(
    dummy_evaluator: DummyEvaluator, dummy_runner: Runner[str, str], dataset_id: str
) -> None:
    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_overview = dummy_evaluator.evaluate_dataset(run_overview.id)

    assert next(iter(evaluation_overview.run_overviews)).dataset_id == dataset_id
    assert evaluation_overview.successful_count == 1
    assert evaluation_overview.failed_count == 2


def test_evaluate_dataset_uses_passed_tracer(
    dummy_evaluator: DummyEvaluator, dataset_id: str, dummy_runner: Runner[str, str]
) -> None:
    in_memory_tracer = InMemoryTracer()
    run_overview = dummy_runner.run_dataset(dataset_id, in_memory_tracer)
    dummy_evaluator.evaluate_dataset(run_overview.id)

    entries = in_memory_tracer.entries
    assert len(entries) == 3
    assert all([isinstance(e, InMemoryTaskSpan) for e in entries])


def test_evaluate_dataset_saves_overview(
    dummy_evaluator: DummyEvaluator,
    dummy_runner: Runner[str, str],
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    dataset_id: str,
) -> None:
    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_overview = dummy_evaluator.evaluate_dataset(run_overview.id)

    assert evaluation_overview == in_memory_evaluation_repository.evaluation_overview(
        evaluation_overview.id,
        EvaluationOverview[DummyAggregatedEvaluationWithResultList],
    )


def test_evaluate_dataset_stores_example_evaluations(
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
    eval_overview = dummy_evaluator.evaluate_dataset(run_overview.id)
    examples = list(dataset)
    success_result = evaluation_repository.example_evaluation(
        eval_overview.id, examples[0].id, DummyEvaluation
    )
    failure_result_task = evaluation_repository.example_evaluation(
        eval_overview.id, examples[1].id, DummyEvaluation
    )
    failure_result_eval = evaluation_repository.example_evaluation(
        eval_overview.id, examples[2].id, DummyEvaluation
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result_task is None
    assert failure_result_eval and isinstance(
        failure_result_eval.result, FailedExampleEvaluation
    )


def test_evaluate_dataset_stores_example_traces(
    dummy_evaluator: DummyEvaluator,
    dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository
    dataset_repository = dummy_evaluator._dataset_repository
    dataset: Optional[Iterable[Example[str, None]]] = dataset_repository.examples_by_id(
        dataset_id, str, type(None)
    )
    assert dataset is not None

    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_run_overview = dummy_evaluator.evaluate_dataset(run_overview.id)
    examples = list(dataset)
    success_result = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], examples[0].id
    )
    failure_result_task = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], examples[1].id
    )
    failure_result_eval = evaluation_repository.example_trace(
        evaluation_run_overview.run_ids[0], examples[2].id
    )

    assert success_result
    assert failure_result_task
    assert failure_result_eval
    assert success_result.trace.input == "success"
    assert failure_result_task.trace.input == FAIL_IN_TASK_INPUT
    assert failure_result_eval.trace.input == FAIL_IN_EVAL_INPUT


def test_evaluate_dataset_stores_aggregated_results(
    dummy_evaluator: DummyEvaluator,
    dummy_runner: Runner[str, str],
    dataset_id: str,
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository

    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_run_overview = dummy_evaluator.evaluate_dataset(run_overview.id)
    loaded_evaluation_run_overview = evaluation_repository.evaluation_overview(
        evaluation_run_overview.id,
        EvaluationOverview[DummyAggregatedEvaluationWithResultList],
    )

    assert evaluation_run_overview == loaded_evaluation_run_overview


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


def test_base_evaluator_type_magic_works(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
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
            *outputs: SuccessfulExampleOutput[Output]
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
        in_memory_evaluation_repository, in_memory_dataset_repository, "dummy"
    )
    who_is_timmy = timmy._get_types()

    assert who_is_timmy == types
