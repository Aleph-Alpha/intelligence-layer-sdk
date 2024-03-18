from typing import Generic, Iterable, Optional, TypeVar

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    InMemoryTaskSpan,
    InMemoryTracer,
    Input,
    NoOpTracer,
    Output,
    Task,
    Tracer,
)
from intelligence_layer.evaluation import (
    AggregatedEvaluation,
    AggregationLogic,
    Aggregator,
    Evaluation,
    EvaluationLogic,
    Evaluator,
    Example,
    ExpectedOutput,
    FailedExampleEvaluation,
    InMemoryAggregationRepository,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    MeanAccumulator,
    Runner,
    SingleOutputEvaluationLogic,
    SuccessfulExampleOutput,
)
from tests.evaluation.conftest import (
    FAIL_IN_EVAL_INPUT,
    FAIL_IN_TASK_INPUT,
    DummyAggregatedEvaluationWithResultList,
    DummyEvaluation,
)


class DummyAggregationLogic(
    AggregationLogic[DummyEvaluation, DummyAggregatedEvaluationWithResultList]
):
    def aggregate(
        self, evaluations: Iterable[DummyEvaluation]
    ) -> DummyAggregatedEvaluationWithResultList:
        return DummyAggregatedEvaluationWithResultList(results=list(evaluations))


class DummyEvaluationLogic(
    SingleOutputEvaluationLogic[
        str,
        str,
        None,
        DummyEvaluation,
    ]
):
    def do_evaluate_single_output(
        self,
        example: Example[str, None],
        output: str,
    ) -> DummyEvaluation:
        if output == FAIL_IN_EVAL_INPUT:
            raise RuntimeError(output)
        return DummyEvaluation(result="pass")


class ComparisonEvaluation(BaseModel):
    is_equal: bool


class ComparisonAggregation(BaseModel):
    equal_ratio: float


class ComparingAggregationLogic(
    AggregationLogic[
        ComparisonEvaluation,
        ComparisonAggregation,
    ]
):
    def aggregate(
        self, evaluations: Iterable[ComparisonEvaluation]
    ) -> ComparisonAggregation:
        acc = MeanAccumulator()
        for evaluation in evaluations:
            acc.add(1.0 if evaluation.is_equal else 0.0)
        return ComparisonAggregation(equal_ratio=acc.extract())


class ComparingEvaluationLogic(
    EvaluationLogic[
        str,
        str,
        None,
        ComparisonEvaluation,
    ]
):
    def do_evaluate(
        self, example: Example[str, None], *output: SuccessfulExampleOutput[str]
    ) -> ComparisonEvaluation:
        unwrapped_output = [o.output for o in output]
        return ComparisonEvaluation(
            is_equal=unwrapped_output[1:] == unwrapped_output[:-1]
        )


class DummyEvaluatorWithoutTypeHints(DummyEvaluationLogic):
    def do_evaluate(self, example: Example, *output: SuccessfulExampleOutput[str]) -> DummyEvaluation:  # type: ignore
        return super().do_evaluate(example, *output)


class DummyTaskWithoutTypeHints(Task[str, str]):
    # type hint for return value missing on purpose for testing
    def do_run(self, input: str, tracer: Tracer):  # type: ignore
        return input


@fixture
def sequence_examples() -> Iterable[Example[str, None]]:
    return [
        Example(input="success", expected_output=None, id="example-1"),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None, id="example-2"),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None, id="example-3"),
    ]


@fixture
def sequence_good_examples() -> Iterable[Example[str, None]]:
    return [
        Example(input="success", expected_output=None),
        Example(input="success", expected_output=None),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None),
    ]


@fixture
def dummy_eval_logic() -> DummyEvaluationLogic:
    return DummyEvaluationLogic()


@fixture
def dummy_aggregate_logic() -> DummyAggregationLogic:
    return DummyAggregationLogic()


@fixture
def dummy_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    dummy_eval_logic: DummyEvaluationLogic,
) -> Evaluator[str, str, None, DummyEvaluation]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "dummy-evaluator",
        dummy_eval_logic,
    )


@fixture
def dummy_aggregator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    dummy_eval_logic: DummyEvaluationLogic,
    dummy_aggregate_logic: DummyAggregationLogic,
) -> Aggregator[DummyEvaluation, DummyAggregatedEvaluationWithResultList]:
    return Aggregator(
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "dummy-evaluator",
        dummy_aggregate_logic,
    )


@fixture
def dataset_id(
    sequence_examples: Iterable[Example[str, None]],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(
        examples=sequence_examples, dataset_name="test-dataset"
    ).id


@fixture
def good_dataset_id(
    sequence_good_examples: Iterable[Example[str, None]],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(
        examples=sequence_good_examples, dataset_name="test-dataset"
    ).id


@fixture
def comparing_eval_logic() -> ComparingEvaluationLogic:
    return ComparingEvaluationLogic()


@fixture
def comparing_aggregation_logic() -> ComparingAggregationLogic:
    return ComparingAggregationLogic()


@fixture
def comparing_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    comparing_eval_logic: ComparingEvaluationLogic,
) -> Evaluator[str, str, None, ComparisonEvaluation]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "comparing-evaluator",
        comparing_eval_logic,
    )


@fixture
def comparing_aggregator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    comparing_aggregation_logic: ComparingAggregationLogic,
) -> Aggregator[ComparisonEvaluation, ComparisonAggregation]:
    return Aggregator(
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "comparing-evaluator",
        comparing_aggregation_logic,
    )


def test_eval_and_aggregate_runs_returns_generic_statistics(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dummy_runner: Runner[str, str],
    dataset_id: str,
) -> None:
    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(evaluation_overview.id)

    assert next(iter(aggregation_overview.run_overviews())).dataset_id == dataset_id
    assert aggregation_overview.successful_evaluation_count == 1
    assert aggregation_overview.failed_evaluation_count == 2


def test_eval_and_aggregate_runs_uses_passed_tracer(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    in_memory_tracer = InMemoryTracer()
    run_overview = dummy_runner.run_dataset(dataset_id, in_memory_tracer)
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)
    dummy_aggregator.aggregate_evaluation(evaluation_overview.id)

    entries = in_memory_tracer.entries
    assert len(entries) == 3
    assert all([isinstance(e, InMemoryTaskSpan) for e in entries])


def test_eval_and_aggregate_runs_stores_example_evaluations(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository
    dataset_repository = dummy_evaluator._dataset_repository
    dataset_id = list(dataset_repository.dataset_ids())[0]
    dataset: Optional[Iterable[Example[str, None]]] = dataset_repository.examples(
        dataset_id, str, type(None)
    )
    assert dataset is not None

    run_overview = dummy_runner.run_dataset(dataset_id, NoOpTracer())
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(evaluation_overview.id)

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
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    run_repository = dummy_evaluator._run_repository
    dataset_repository = dummy_evaluator._dataset_repository
    dataset: Optional[Iterable[Example[str, None]]] = dataset_repository.examples(
        dataset_id, str, type(None)
    )
    assert dataset is not None

    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(evaluation_overview.id)

    examples = list(dataset)
    success_result = run_repository.example_trace(
        aggregation_overview.run_ids[0], examples[0].id
    )
    failure_result_task = run_repository.example_trace(
        aggregation_overview.run_ids[0], examples[1].id
    )
    failure_result_eval = run_repository.example_trace(
        aggregation_overview.run_ids[0], examples[2].id
    )

    assert success_result
    assert failure_result_task
    assert failure_result_eval
    assert success_result.trace.input == "success"
    assert failure_result_task.trace.input == FAIL_IN_TASK_INPUT
    assert failure_result_eval.trace.input == FAIL_IN_EVAL_INPUT


def test_eval_and_aggregate_runs_stores_aggregated_results(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dummy_runner: Runner[str, str],
    dataset_id: str,
) -> None:
    aggregation_repository = dummy_aggregator._aggregation_repository

    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(evaluation_overview.id)
    loaded_evaluation_run_overview = aggregation_repository.aggregation_overview(
        aggregation_overview.id, DummyAggregatedEvaluationWithResultList
    )

    assert aggregation_overview == loaded_evaluation_run_overview


def test_evaluate_can_evaluate_multiple_runs(
    comparing_evaluator: Evaluator[str, str, None, ComparisonEvaluation],
    comparing_aggregator: Aggregator[ComparisonEvaluation, ComparisonAggregation],
    string_dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    run_overview1 = dummy_runner.run_dataset(string_dataset_id)
    run_overview2 = dummy_runner.run_dataset(string_dataset_id)

    evaluation_overview = comparing_evaluator.evaluate_runs(
        run_overview1.id, run_overview2.id
    )
    aggregation_overview = comparing_aggregator.aggregate_evaluation(
        evaluation_overview.id
    )
    assert aggregation_overview.statistics.equal_ratio == 1


def test_aggregate_evaluation_can_aggregate_multiple_evals(
    comparing_evaluator: Evaluator[str, str, None, ComparisonEvaluation],
    comparing_aggregator: Aggregator[ComparisonEvaluation, ComparisonAggregation],
    string_dataset_id: str,
    dummy_runner: Runner[str, str],
) -> None:
    run_overview_1 = dummy_runner.run_dataset(string_dataset_id)
    run_overview_2 = dummy_runner.run_dataset(string_dataset_id)

    evaluation_overview_1 = comparing_evaluator.evaluate_runs(run_overview_1.id)
    evaluation_overview_2 = comparing_evaluator.evaluate_runs(
        run_overview_1.id, run_overview_2.id
    )

    aggregation_overview = comparing_aggregator.aggregate_evaluation(
        evaluation_overview_1.id, evaluation_overview_1.id, evaluation_overview_2.id
    )

    assert len(list(aggregation_overview.run_overviews())) == 2
    assert aggregation_overview.statistics.equal_ratio == 1


def test_evaluator_type_magic_works(
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
    }

    A = TypeVar("A", bound=BaseModel)

    class Mailman(Generic[A]):
        pass

    class ChildEvaluationLogic(EvaluationLogic[Input, Output, None, Evaluation]):
        def do_evaluate(
            self,
            example: Example[Input, ExpectedOutput],
            *output: SuccessfulExampleOutput[Output],
        ) -> Evaluation:
            return None  # type: ignore

    class ChildAggregationLogic(AggregationLogic[Evaluation, AggregatedEvaluation]):
        def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
            return None  # type: ignore

    class GrandChildEvaluationLogic(ChildEvaluationLogic[Input, str, A]):
        pass

    class GreatGrandChildEvaluationLogic(
        Mailman[EvaluationType], GrandChildEvaluationLogic[str, EvaluationType]
    ):
        pass

    timmy: Evaluator[
        str,
        str,
        None,
        EvaluationType,
    ] = Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "dummy",
        evaluation_logic=GreatGrandChildEvaluationLogic(),
    )
    who_is_timmy = timmy._get_types()

    assert who_is_timmy == types


def test_aggregator_type_magic_works(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
) -> None:
    class EvaluationType(BaseModel):
        pass

    class AggregatedEvaluationType(BaseModel):
        pass

    types = {
        "Evaluation": EvaluationType,
        "AggregatedEvaluation": AggregatedEvaluationType,
    }

    class ChildAggregationLogic(AggregationLogic[Evaluation, AggregatedEvaluationType]):
        def aggregate(
            self, evaluations: Iterable[Evaluation]
        ) -> AggregatedEvaluationType:
            return None  # type: ignore

    class GrandChildAggregationLogic(ChildAggregationLogic[EvaluationType]):
        pass

    timmy: Aggregator[EvaluationType, AggregatedEvaluationType] = Aggregator(
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "dummy",
        aggregation_logic=GrandChildAggregationLogic(),
    )
    who_is_timmy = timmy._get_types()

    assert who_is_timmy == types


def test_eval_and_aggregate_runs_only_runs_n_examples(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dummy_runner: Runner[str, str],
    good_dataset_id: str,
) -> None:
    run_overview = dummy_runner.run_dataset(good_dataset_id)
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(evaluation_overview.id)

    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id, num_examples=2)
    aggregation_overview_n = dummy_aggregator.aggregate_evaluation(
        evaluation_overview.id
    )

    assert (
        aggregation_overview.successful_evaluation_count
        + aggregation_overview.crashed_during_evaluation_count
        == 3
    )
    assert (
        aggregation_overview_n.successful_evaluation_count
        + aggregation_overview_n.crashed_during_evaluation_count
        == 2
    )
