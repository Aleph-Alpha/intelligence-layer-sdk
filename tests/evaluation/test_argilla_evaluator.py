import random
from typing import Iterable, Sequence, cast
from uuid import uuid4

from pytest import fixture

from intelligence_layer.connectors import ArgillaEvaluation, Field, Question, RecordData
from intelligence_layer.connectors.argilla.argilla_client import ArgillaClient
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
from tests.conftest import (
    DummyStringEvaluation,
    DummyStringInput,
    DummyStringOutput,
    DummyStringTask,
)
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
        DummyStringEvaluation,
    ]
):
    def __init__(self) -> None:
        super().__init__(
            fields={
                "output": Field(name="output", title="Output"),
                "input": Field(name="input", title="Input"),
            },
            questions=[
                Question(
                    name="name", title="title", description="description", options=[0]
                )
            ],
        )

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

    def _from_record(
        self, argilla_evaluation: ArgillaEvaluation
    ) -> DummyStringEvaluation:
        return DummyStringEvaluation()


class DummyArgillaClient(ArgillaClient):
    _datasets: dict[str, list[RecordData]] = {}
    _score = 3.0

    def ensure_dataset_exists(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        dataset_id = str(uuid4())
        self._datasets[dataset_id] = []
        return dataset_id

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        if dataset_id not in self._datasets.keys():
            raise Exception("Add record: dataset not found")
        self._datasets[dataset_id].append(record)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        dataset = self._datasets.get(dataset_id)
        assert dataset
        return [
            ArgillaEvaluation(
                example_id=record.example_id,
                record_id="ignored",
                responses={"human-score": self._score},
                metadata=dict(),
            )
            for record in dataset
        ]

    def split_dataset(self, dataset_id: str, n_splits: int) -> None:
        raise NotImplementedError


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
    DummyStringEvaluation,
]:
    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        argilla_evaluation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaEvaluationLogic(),
        StubArgillaClient(),
        "workspace-id",
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


def test_argilla_evaluator_can_submit_evals_to_argilla(
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
) -> None:
    # fetch run_overview
    # put run_oervrw ionto submit function
    # check if stuff is correctly submitted -> ArgillaStubClient has receveid data

    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaEvaluationLogic(),
        DummyArgillaClient(),
        workspace_id="workspace-id",
    )

    run_overview = string_argilla_runner.run_dataset(string_dataset_id)

    partial_evaluation_overview = evaluator.submit(run_overview.id)

    eval_overview = evaluator.retrieve(partial_evaluation_overview.id)

    assert eval_overview.end_date is not None
    assert eval_overview.successful_evaluation_count == 1
    assert eval_overview.failed_evaluation_count == 0

    assert (
        len(
            in_memory_evaluation_repository.example_evaluations(
                eval_overview.id, DummyStringOutput
            )
        )
        == 1
    )

    assert len(list(in_memory_evaluation_repository.evaluation_overviews())) == 1
    assert len(DummyArgillaClient()._datasets[partial_evaluation_overview.id]) == 1


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
