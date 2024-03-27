import random
from typing import Iterable, Sequence, cast

from pytest import fixture

from intelligence_layer.connectors import ArgillaEvaluation, Field, Question, RecordData
from intelligence_layer.evaluation import (
    AggregationLogic,
    ArgillaAggregator,
    ArgillaEvaluationLogic,
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    Example,
    InMemoryAggregationRepository,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    InstructComparisonArgillaAggregationLogic,
    RecordDataSequence,
    Runner,
    SuccessfulExampleOutput,
)
from tests.conftest import DummyStringInput, DummyStringOutput, DummyStringTask
from tests.evaluation.conftest import DummyAggregatedEvaluation, StubArgillaClient


class DummyStringTaskArgillaAggregationLogic(
    AggregationLogic[
        ArgillaEvaluation,
        DummyAggregatedEvaluation,
    ]
):
    def aggregate(
        self,
        evaluations: Iterable[ArgillaEvaluation],
    ) -> DummyAggregatedEvaluation:
        evaluations = list(evaluations)
        total_human_score = sum(
            cast(float, a.responses["human-score"]) for a in evaluations
        )
        return DummyAggregatedEvaluation(
            score=total_human_score / len(evaluations),
        )


class DummyStringTaskArgillaEvaluationLogic(
    ArgillaEvaluationLogic[
        DummyStringInput,
        DummyStringOutput,
        DummyStringOutput,
    ]
):
    def _to_record(
        self,
        example: Example[DummyStringInput, DummyStringOutput],
        *output: SuccessfulExampleOutput[DummyStringOutput],
    ) -> RecordDataSequence:
        assert len(output) == 1
        single_output = output[0].output
        return RecordDataSequence(
            records=[
                RecordData(
                    content={
                        "input": example.input.input,
                        "output": single_output.output,
                    },
                    example_id=example.id,
                )
            ]
        )


@fixture
def arg() -> StubArgillaClient:
    return StubArgillaClient()


@fixture()
def argilla_questions() -> Sequence[Question]:
    return [
        Question(
            name="question",
            title="title",
            description="description",
            options=[1],
        )
    ]


@fixture()
def argilla_fields() -> Sequence[Field]:
    return [
        Field(name="output", title="Output"),
        Field(name="input", title="Input"),
    ]


@fixture
def argilla_evaluation_repository(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    stub_argilla_client: StubArgillaClient,
    argilla_questions: Sequence[Question],
    argilla_fields: Sequence[Field],
) -> ArgillaEvaluationRepository:
    stub_argilla_client._expected_workspace_id = "workspace-id"
    stub_argilla_client._expected_questions = argilla_questions
    stub_argilla_client._expected_fields = argilla_fields

    workspace_id = stub_argilla_client._expected_workspace_id

    return ArgillaEvaluationRepository(
        in_memory_evaluation_repository,
        stub_argilla_client,
        workspace_id,
        argilla_fields,
        argilla_questions,
    )


@fixture
def string_argilla_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    argilla_evaluation_repository: ArgillaEvaluationRepository,
) -> ArgillaEvaluator[
    DummyStringInput,
    DummyStringOutput,
    DummyStringOutput,
]:
    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        argilla_evaluation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaEvaluationLogic(),
    )
    return evaluator


@fixture
def string_argilla_aggregator(
    argilla_evaluation_repository: ArgillaEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
) -> ArgillaAggregator[DummyAggregatedEvaluation]:
    aggregator = ArgillaAggregator(
        argilla_evaluation_repository,
        in_memory_aggregation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaAggregationLogic(),
    )
    return aggregator


@fixture
def string_argilla_runner(
    dummy_string_task: DummyStringTask,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> Runner[DummyStringInput, DummyStringOutput]:
    return Runner(
        dummy_string_task,
        in_memory_dataset_repository,
        in_memory_run_repository,
        "dummy-task",
    )


def test_argilla_evaluator_can_do_sync_evaluation(
    string_argilla_evaluator: ArgillaEvaluator[
        DummyStringInput,
        DummyStringOutput,
        DummyStringOutput,
    ],
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
) -> None:
    argilla_client = cast(
        StubArgillaClient,
        string_argilla_evaluator._evaluation_repository._client,  # type: ignore
    )

    run_overview = string_argilla_runner.run_dataset(string_dataset_id)
    eval_overview = string_argilla_evaluator.evaluate_runs(run_overview.id)
    examples_iter = string_argilla_evaluator._dataset_repository.examples(
        string_dataset_id, DummyStringInput, DummyStringOutput
    )
    assert examples_iter is not None

    assert eval_overview.id in argilla_client._datasets
    saved_dataset = argilla_client._datasets[eval_overview.id]
    examples = list(examples_iter)
    assert len(saved_dataset) == len(examples)
    assert saved_dataset[0].example_id == examples[0].id
    assert saved_dataset[0].content["input"] == examples[0].input.input


def test_argilla_evaluator_can_aggregate_evaluation(
    string_argilla_evaluator: ArgillaEvaluator[
        DummyStringInput,
        DummyStringOutput,
        DummyStringOutput,
    ],
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
    string_argilla_aggregator: ArgillaAggregator[DummyAggregatedEvaluation],
) -> None:
    # given
    argilla_client = cast(
        StubArgillaClient, string_argilla_evaluator._evaluation_repository._client  # type: ignore
    )
    # when
    run_overview = string_argilla_runner.run_dataset(string_dataset_id)
    eval_overview = string_argilla_evaluator.evaluate_runs(run_overview.id)
    aggregated_eval_overview = string_argilla_aggregator.aggregate_evaluation(
        eval_overview.id
    )
    # then
    assert aggregated_eval_overview.statistics.score == argilla_client._score


def test_argilla_aggregation_logic_works() -> None:
    argilla_aggregation_logic = InstructComparisonArgillaAggregationLogic()
    evaluations = (
        ArgillaEvaluation(
            example_id=str(i),
            record_id=str(i),
            responses={"winner": random.choices([1, 2, 3], [0.5, 0.25, 0.25], k=1)[0]},
            metadata={
                "first": "player_1",
                "second": "player_2" if i < 9000 else "player_3",
            },
        )
        for i in range(10000)
    )
    aggregation = argilla_aggregation_logic.aggregate(evaluations)
    assert aggregation.scores["player_1"].elo > aggregation.scores["player_2"].elo
    assert (
        aggregation.scores["player_3"].elo_standard_error
        > aggregation.scores["player_1"].elo_standard_error
    )
