from collections.abc import Iterable
from typing import Generic, Optional, TypeVar

import pytest
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import Input, Output, Task, Tracer
from intelligence_layer.core.tracer.in_memory_tracer import (
    InMemoryTaskSpan,
    InMemoryTracer,
)
from intelligence_layer.evaluation import (
    AggregatedEvaluation,
    AggregationLogic,
    Aggregator,
    Evaluation,
    EvaluationLogic,
    Evaluator,
    Example,
    ExampleOutput,
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


class DummyPairwiseEvaluationLogic(
    EvaluationLogic[
        str,
        str,
        None,
        DummyEvaluation,
    ]
):
    def do_evaluate(
        self,
        example: Example[str, None],
        *output: SuccessfulExampleOutput[str],
    ) -> DummyEvaluation:
        for out in output:
            if out.output == FAIL_IN_EVAL_INPUT:
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
    def do_evaluate(  # type: ignore
        self, example: Example[str, None], *output: SuccessfulExampleOutput[str]
    ) -> DummyEvaluation:
        return super().do_evaluate(example, *output)


class DummyTaskWithoutTypeHints(Task[str, str]):
    # type hint for return value missing on purpose for testing
    def do_run(self, input: str, tracer: Tracer):  # type: ignore
        return input


@fixture
def sequence_good_examples() -> Iterable[Example[str, None]]:
    return [
        Example(input="success", expected_output=None, id="0"),
        Example(input="success", expected_output=None, id="1"),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None, id="2"),
    ]


@fixture
def dummy_eval_logic() -> DummyEvaluationLogic:
    return DummyEvaluationLogic()


@fixture
def dummy_pairwise_eval_logic() -> DummyPairwiseEvaluationLogic:
    return DummyPairwiseEvaluationLogic()


@fixture
def dummy_aggregate_logic() -> DummyAggregationLogic:
    return DummyAggregationLogic()


class SuccessfulDummyTask(Task[str, str]):
    def do_run(self, input: str, tracer: Tracer) -> str:
        return input


@fixture
def successful_dummy_runner(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> Runner[str, str]:
    return Runner(
        SuccessfulDummyTask(),
        in_memory_dataset_repository,
        in_memory_run_repository,
        "successful-dummy-runner",
    )


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
def dummy_pairwise_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    dummy_pairwise_eval_logic: DummyPairwiseEvaluationLogic,
) -> Evaluator[str, str, None, DummyEvaluation]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "dummy-evaluator",
        dummy_pairwise_eval_logic,
    )


@fixture
def dummy_aggregator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
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
def run_id(
    dataset_id: str,
    dummy_runner: Runner[str, str],
) -> str:
    return dummy_runner.run_dataset(dataset_id).id


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


def test_eval_runs_returns_generic_statistics(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_runner: Runner[str, str],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> None:
    examples = [
        Example(input="success", expected_output=None, id="example-1"),
        Example(input="success", expected_output=None, id="example-2"),
        Example(input="success", expected_output=None, id="example-3"),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None, id="example-4"),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None, id="example-5"),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None, id="example-6"),
    ]
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name="test-dataset"
    ).id

    run_overview = dummy_runner.run_dataset(dataset_id)
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)

    assert evaluation_overview.successful_evaluation_count == 3
    assert evaluation_overview.failed_evaluation_count == 1


def test_eval_runs_works_without_description(
    run_id: str,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    dummy_eval_logic: DummyEvaluationLogic,
) -> None:
    evaluator = Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "",
        dummy_eval_logic,
    )
    evaluation_overview = evaluator.evaluate_runs(run_id)

    assert evaluation_overview.description == evaluator.description


def test_eval_runs_uses_correct_description(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation], run_id: str
) -> None:
    eval_description = "My evaluation description"
    evaluation_overview = dummy_evaluator.evaluate_runs(
        run_id, description=eval_description
    )

    assert dummy_evaluator.description in evaluation_overview.description
    assert eval_description in evaluation_overview.description


def test_eval_runs_keeps_example_for_eval_if_skip_flag_is_false(
    dummy_pairwise_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_runner: Runner[str, str],
    successful_dummy_runner: Runner[str, str],
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> None:
    examples = [
        Example(input="success", expected_output=None, id="example-1"),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None, id="example-2"),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None, id="example-3"),
    ]
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name="test-dataset"
    ).id

    run_overview_with_failure = dummy_runner.run_dataset(dataset_id)
    successful_run_overview = successful_dummy_runner.run_dataset(dataset_id)

    evaluation_overview = dummy_pairwise_evaluator.evaluate_runs(
        run_overview_with_failure.id,
        successful_run_overview.id,
        skip_example_on_any_failure=False,
    )

    assert evaluation_overview.successful_evaluation_count == 2
    assert evaluation_overview.failed_evaluation_count == 1


def test_evaluator_aborts_on_error(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    run_id: str,
) -> None:
    with pytest.raises(RuntimeError):
        dummy_evaluator.evaluate_runs(run_id, abort_on_error=True)


def test_eval_and_aggregate_runs_stores_example_evaluations(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dataset_id: str,
    run_id: str,
) -> None:
    evaluation_repository = dummy_evaluator._evaluation_repository
    dataset_repository = dummy_evaluator._dataset_repository
    examples = list(dataset_repository.examples(dataset_id, str, type(None)))

    evaluation_overview = dummy_evaluator.evaluate_runs(run_id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(evaluation_overview.id)
    assert next(iter(aggregation_overview.evaluation_overviews)) == evaluation_overview

    success_result = evaluation_repository.example_evaluation(
        evaluation_overview.id,
        examples[0].id,
        DummyEvaluation,
    )
    failure_result_task = evaluation_repository.example_evaluation(
        evaluation_overview.id,
        examples[1].id,
        DummyEvaluation,
    )
    failure_result_eval = evaluation_repository.example_evaluation(
        evaluation_overview.id,
        examples[2].id,
        DummyEvaluation,
    )

    assert success_result and isinstance(success_result.result, DummyEvaluation)
    assert failure_result_task is None
    assert failure_result_eval and isinstance(
        failure_result_eval.result, FailedExampleEvaluation
    )


def test_failed_evaluations_returns_only_failed_evaluations(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dataset_id: str,
    run_id: str,
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    evaluation_overview = dummy_evaluator.evaluate_runs(run_id)
    failed_evaluations = list(
        dummy_evaluator.failed_evaluations(evaluation_overview.id)
    )

    assert len(failed_evaluations) == 1
    assert isinstance(failed_evaluations[0].evaluation.result, FailedExampleEvaluation)
    assert failed_evaluations[0].example.id == list(sequence_examples)[-1].id


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
    success_result = run_repository.example_tracer(
        aggregation_overview.run_ids[0], examples[0].id
    )

    failure_result_task = run_repository.example_tracer(
        aggregation_overview.run_ids[0], examples[1].id
    )
    failure_result_eval = run_repository.example_tracer(
        aggregation_overview.run_ids[0], examples[2].id
    )

    assert success_result
    assert type(success_result) is InMemoryTracer
    assert failure_result_task
    assert type(failure_result_task) is InMemoryTracer
    assert failure_result_eval
    assert type(failure_result_eval) is InMemoryTracer

    assert type(success_result.entries[0]) is InMemoryTaskSpan
    assert success_result.entries[0].input == "success"

    assert type(failure_result_task.entries[0]) is InMemoryTaskSpan
    assert failure_result_task.entries[0].input == FAIL_IN_TASK_INPUT

    assert type(failure_result_eval.entries[0]) is InMemoryTaskSpan
    assert failure_result_eval.entries[0].input == FAIL_IN_EVAL_INPUT


def test_eval_and_aggregate_runs_stores_aggregated_results(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    dataset_id: str,
    run_id: str,
) -> None:
    aggregation_repository = dummy_aggregator._aggregation_repository

    evaluation_overview = dummy_evaluator.evaluate_runs(run_id)
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
    who_is_timmy = timmy._get_types

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
    who_is_timmy = timmy._get_types

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


def test_eval_works_when_runs_are_not_complete(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_runner: Runner[str, str],
    good_dataset_id: str,
) -> None:
    run_overview = dummy_runner.run_dataset(good_dataset_id, num_examples=2)
    evaluation_overview = dummy_evaluator.evaluate_runs(run_overview.id)
    assert evaluation_overview.successful_evaluation_count == 2


def test_eval_raises_errors_when_runs_are_not_the_same_length(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_runner: Runner[str, str],
    good_dataset_id: str,
) -> None:
    run_1 = dummy_runner.run_dataset(good_dataset_id, num_examples=2)
    run_2 = dummy_runner.run_dataset(good_dataset_id, num_examples=1)
    with pytest.raises(ValueError, match="number"):
        dummy_evaluator.evaluate_runs(run_1.id, run_2.id)


def test_eval_raises_error_if_examples_and_example_outputs_dont_match(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
) -> None:
    examples = [Example(input="", expected_output=None)]
    not_matching_example_outputs = [
        (ExampleOutput(run_id="id", example_id="not-matching-id", output="output"),)
    ]
    with pytest.raises(ValueError):
        list(
            dummy_evaluator._generate_evaluation_inputs(
                examples,
                not_matching_example_outputs,
                skip_example_on_any_failure=False,
                num_examples=None,
            )
        )


def test_evaluator_evaluate_runs_sets_default_values(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation], run_id: str
) -> None:
    evaluation_overview = dummy_evaluator.evaluate_runs(run_id)
    assert evaluation_overview.labels == set()
    assert evaluation_overview.metadata == dict()


def test_evaluator_evaluate_runs_specific_values_overwrite_defaults(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation], run_id: str
) -> None:
    expected_labels = {"test_label"}
    expected_metadata: SerializableDict = dict({"test_key": "test-value"})
    evaluation_overview = dummy_evaluator.evaluate_runs(
        run_id, labels=expected_labels, metadata=expected_metadata
    )
    assert evaluation_overview.labels == expected_labels
    assert evaluation_overview.metadata == expected_metadata


def test_aggregate_evaluation_set_default_labels_metadata_values(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    run_id: str,
) -> None:
    evaluation_overview = dummy_evaluator.evaluate_runs(run_id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(evaluation_overview.id)

    assert aggregation_overview.labels == set()
    assert aggregation_overview.metadata == dict()


def test_aggregate_evaluation_specific_values_overwrite_defaults(
    dummy_evaluator: Evaluator[str, str, None, DummyEvaluation],
    dummy_aggregator: Aggregator[
        DummyEvaluation, DummyAggregatedEvaluationWithResultList
    ],
    run_id: str,
) -> None:
    expected_labels = {"test_label"}
    expected_metadata: SerializableDict = dict({"test_key": "test-value"})
    evaluation_overview = dummy_evaluator.evaluate_runs(run_id)
    aggregation_overview = dummy_aggregator.aggregate_evaluation(
        evaluation_overview.id, labels=expected_labels, metadata=expected_metadata
    )

    assert aggregation_overview.labels == expected_labels
    assert aggregation_overview.metadata == expected_metadata
