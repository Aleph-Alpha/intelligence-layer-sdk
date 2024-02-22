from typing import Iterable, Sequence, cast
from uuid import uuid4

from pytest import fixture

from intelligence_layer.connectors import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.evaluation import (
    ArgillaEvaluationLogic,
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    RecordDataSequence,
    Runner,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.argilla import ArgillaAggregator
from intelligence_layer.evaluation.base_logic import AggregationLogic
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    InMemoryAggregationRepository,
)
from intelligence_layer.evaluation.data_storage.run_repository import (
    InMemoryRunRepository,
)
from tests.conftest import DummyStringInput, DummyStringOutput, DummyStringTask
from tests.evaluation.conftest import DummyAggregatedEvaluation


class StubArgillaClient(ArgillaClient):
    _expected_workspace_id: str
    _expected_fields: Sequence[Field]
    _expected_questions: Sequence[Question]
    _datasets: dict[str, list[RecordData]] = {}
    _score = 3.0

    def create_dataset(
        self,
        workspace_id: str,
        _: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        if workspace_id != self._expected_workspace_id:
            raise Exception("Incorrect workspace id")
        elif fields != self._expected_fields:
            raise Exception("Incorrect fields")
        elif questions != self._expected_questions:
            raise Exception("Incorrect questions")
        id = str(uuid4())
        self._datasets[id] = []
        return id

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        if dataset_id not in self._datasets:
            raise Exception("Add record: dataset not found")
        self._datasets[dataset_id].append(record)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        dataset = self._datasets.get(dataset_id)
        assert dataset
        return [
            ArgillaEvaluation(
                example_id="something",
                record_id="ignored",
                responses={"human-score": self._score},
                metadata=dict(),
            )
            for _ in dataset
        ]


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
def stub_argilla_client() -> StubArgillaClient:
    return StubArgillaClient()


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
    stub_argilla_client: StubArgillaClient,
    argilla_questions: Sequence[Question],
    argilla_fields: Sequence[Field],
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
    stub_argilla_client: StubArgillaClient,
    argilla_questions: Sequence[Question],
    argilla_fields: Sequence[Field],
) -> ArgillaAggregator[DummyAggregatedEvaluation,]:
    workspace_id = stub_argilla_client._expected_workspace_id

    eval_repository = ArgillaEvaluationRepository(
        argilla_evaluation_repository,
        stub_argilla_client,
        workspace_id,
        argilla_fields,
        argilla_questions,
    )

    evaluator = ArgillaAggregator(
        eval_repository,
        in_memory_aggregation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaAggregationLogic(),
    )
    return evaluator


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
    examples_iter = string_argilla_evaluator._dataset_repository.examples_by_id(
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
    string_argilla_aggregator: ArgillaAggregator[DummyAggregatedEvaluation],
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
) -> None:
    argilla_client = cast(
        StubArgillaClient, string_argilla_evaluator._evaluation_repository._client  # type: ignore
    )
    run_overview = string_argilla_runner.run_dataset(string_dataset_id)
    eval_overview = string_argilla_evaluator.evaluate_runs(run_overview.id)
    aggregated_eval_overview = string_argilla_aggregator.aggregate_evaluation(
        eval_overview.id
    )
    assert aggregated_eval_overview.statistics.score == argilla_client._score
