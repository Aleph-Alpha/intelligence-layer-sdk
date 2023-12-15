from typing import Generic, Iterable, Optional, TypeVar

from pydantic import BaseModel
from pytest import fixture, raises

from intelligence_layer.core import (
    Evaluator,
    Example,
    FailedExampleEvaluation,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryTaskSpan,
    InMemoryTracer,
    NoOpTracer,
    SequenceDataset,
    Tracer,
)
from intelligence_layer.core.evaluation.domain import (
    Dataset,
    Evaluation,
    EvaluationOverview,
    ExpectedOutput,
)
from intelligence_layer.core.evaluation.evaluator import BaseEvaluator
from intelligence_layer.core.task import Input, Output, Task
from tests.core.evaluation.conftest import (
    DummyAggregatedEvaluationWithResultList,
    DummyEvaluation,
)

FAIL_IN_EVAL_INPUT = "fail in eval"
FAIL_IN_TASK_INPUT = "fail in task"


class DummyEvaluator(
    Evaluator[
        str,
        str,
        None,
        DummyEvaluation,
        DummyAggregatedEvaluationWithResultList,
    ]
):
    def expected_output_type(self) -> type[None]:
        return type(None)

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
    def expected_output_type(self) -> type[None]:
        return type(None)

    def do_evaluate(
        self, input: str, expected_output: None, *output: str
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
        self, input: str, output: str, expected_output: None
    ):
        return super().do_evaluate(input, expected_output, output)


class DummyTask(Task[str, str]):
    def do_run(self, input: str, tracer: Tracer) -> str:
        if input == FAIL_IN_TASK_INPUT:
            raise RuntimeError(input)
        return input


@fixture
def sequence_dataset() -> SequenceDataset[str, None]:
    examples = [
        Example(input="success", expected_output=None),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None),
    ]
    return SequenceDataset(name="test", examples=examples)


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
    sequence_dataset: SequenceDataset[str, None],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    in_memory_dataset_repository.create_dataset(
        sequence_dataset.name, sequence_dataset.examples
    )
    return sequence_dataset.name


@fixture
def comparing_evaluator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> ComparingEvaluator:
    return ComparingEvaluator(
        DummyTask(), in_memory_evaluation_repository, in_memory_dataset_repository
    )


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
    dummy_evaluator: DummyEvaluator,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    dataset_name: str,
) -> None:
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
    dataset_name = list(dataset_repository.list_datasets())[0]
    dataset: Optional[Dataset[str, None]] = dataset_repository.dataset(
        dataset_name, str, type(None)
    )
    assert dataset

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(
        dataset_name, NoOpTracer()
    )
    examples = list(dataset.examples)
    success_result = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, examples[0].id, DummyEvaluation
    )
    failure_result_task = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, examples[1].id, DummyEvaluation
    )
    failure_result_eval = evaluation_repository.example_evaluation(
        evaluation_run_overview.id, examples[2].id, DummyEvaluation
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
    dataset: Optional[Dataset[str, None]] = dataset_repository.dataset(
        dataset_name, str, type(None)
    )
    assert dataset

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset_name)
    examples = list(dataset.examples)
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
    dataset_name: str,
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset_name)
    loaded_evaluation_run_overview = evaluation_repository.evaluation_overview(
        evaluation_run_overview.id,
        EvaluationOverview[DummyAggregatedEvaluationWithResultList],
    )

    assert evaluation_run_overview == loaded_evaluation_run_overview


def test_evaluate_can_evaluate_multiple_runs(
    dummy_evaluator: DummyEvaluator,
    comparing_evaluator: ComparingEvaluator,
    string_dataset_name: str,
) -> None:
    run_overview1 = dummy_evaluator.run_dataset(string_dataset_name)
    run_overview2 = dummy_evaluator.run_dataset(string_dataset_name)

    partial_overview = comparing_evaluator.evaluate_run(
        run_overview1.id, run_overview2.id
    )

    eval_overview = comparing_evaluator.aggregate_evaluation(partial_overview.id)
    assert eval_overview.statistics.equal_ratio == 1


def test_evaluation_type_raises_if_do_evaluate_does_not_have_type_hints() -> None:
    dummy_evaluator = DummyEvaluatorWithoutTypeHints(
        DummyTask(), InMemoryEvaluationRepository(), InMemoryDatasetRepository()
    )

    with raises(TypeError):
        dummy_evaluator.evaluation_type()


def test_base_evaluator_type_magic_works(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> None:
    class EvaluationType(BaseModel):
        pass

    class AggregatedEvaluationType(BaseModel):
        pass

    types = [
        str,
        str,
        type(None),
        EvaluationType,
        AggregatedEvaluationType,
    ]

    class ChildEvaluator(
        BaseEvaluator[Input, Output, None, Evaluation, AggregatedEvaluationType]
    ):
        def evaluate(
            self, example: Example[Input, ExpectedOutput], eval_id: str, *output: Output
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
        DummyTask(), in_memory_evaluation_repository, in_memory_dataset_repository
    )
    who_is_timmy = timmy._get_types()

    assert who_is_timmy == types
