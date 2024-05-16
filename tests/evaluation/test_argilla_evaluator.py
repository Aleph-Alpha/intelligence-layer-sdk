import random
from typing import Iterable, Sequence
from uuid import uuid4

import pytest
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
    ArgillaEvaluator,
    AsyncInMemoryEvaluationRepository,
    ComparisonEvaluation,
    ComparisonEvaluationAggregationLogic,
    DatasetRepository,
    Example,
    InMemoryDatasetRepository,
    InMemoryRunRepository,
    MatchOutcome,
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
from tests.evaluation.conftest import StubArgillaClient


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

    def to_record(
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

    def from_record(
        self, argilla_evaluation: ArgillaEvaluation
    ) -> DummyStringEvaluation:
        return DummyStringEvaluation()


class CustomException(Exception):
    pass


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


class FailedEvaluationDummyArgillaClient(ArgillaClient):
    """fails on first upload, only returns 1 evaluated evaluation"""

    _upload_count = 0
    _datasets: dict[str, list[RecordData]] = {}

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
        if self._upload_count == 0:
            self._upload_count += 1
            raise CustomException("First upload fails")
        self._datasets[dataset_id].append(record)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        dataset = self._datasets.get(dataset_id)
        assert dataset
        record = dataset[0]
        return [
            ArgillaEvaluation(
                example_id=record.example_id,
                record_id="ignored",
                responses={"human-score": 0},
                metadata=dict(),
            )
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
def string_argilla_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    async_in_memory_evaluation_repository: AsyncInMemoryEvaluationRepository,
) -> ArgillaEvaluator[
    DummyStringInput,
    DummyStringOutput,
    DummyStringOutput,
    DummyStringEvaluation,
]:
    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        async_in_memory_evaluation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaEvaluationLogic(),
        StubArgillaClient(),
        "workspace-id",
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


def test_argilla_evaluator_can_submit_evals_to_argilla(
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    async_in_memory_evaluation_repository: AsyncInMemoryEvaluationRepository,
) -> None:
    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        async_in_memory_evaluation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaEvaluationLogic(),
        DummyArgillaClient(),
        workspace_id="workspace-id",
    )

    run_overview = string_argilla_runner.run_dataset(string_dataset_id)

    partial_evaluation_overview = evaluator.submit(run_overview.id)
    assert partial_evaluation_overview.submitted_evaluation_count == 1

    eval_overview = evaluator.retrieve(partial_evaluation_overview.id)

    assert eval_overview.end_date is not None
    assert eval_overview.successful_evaluation_count == 1
    assert eval_overview.failed_evaluation_count == 0

    assert (
        len(
            async_in_memory_evaluation_repository.example_evaluations(
                eval_overview.id, DummyStringOutput
            )
        )
        == 1
    )

    assert len(list(async_in_memory_evaluation_repository.evaluation_overviews())) == 1
    assert len(DummyArgillaClient()._datasets[partial_evaluation_overview.id]) == 1


def test_argilla_evaluator_correctly_lists_failed_eval_counts(
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    async_in_memory_evaluation_repository: AsyncInMemoryEvaluationRepository,
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
) -> None:
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[dummy_string_example] * 3, dataset_name="test-dataset"
    ).id
    run_overview = string_argilla_runner.run_dataset(dataset_id)

    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        async_in_memory_evaluation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaEvaluationLogic(),
        FailedEvaluationDummyArgillaClient(),
        workspace_id="workspace-id",
    )

    partial_evaluation_overview = evaluator.submit(run_overview.id)
    assert (
        len(
            async_in_memory_evaluation_repository.failed_example_evaluations(
                partial_evaluation_overview.id, DummyStringEvaluation
            )
        )
        == 1
    )
    eval_overview = evaluator.retrieve(partial_evaluation_overview.id)

    assert eval_overview.successful_evaluation_count == 1
    assert eval_overview.failed_evaluation_count == 2


def test_argilla_evaluator_abort_on_error_works(
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    async_in_memory_evaluation_repository: AsyncInMemoryEvaluationRepository,
) -> None:
    run_overview = string_argilla_runner.run_dataset(string_dataset_id)

    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        async_in_memory_evaluation_repository,
        "dummy-string-task",
        DummyStringTaskArgillaEvaluationLogic(),
        FailedEvaluationDummyArgillaClient(),
        workspace_id="workspace-id",
    )
    with pytest.raises(CustomException):
        evaluator.submit(run_overview.id, abort_on_error=True)


def test_argilla_aggregation_logic_works() -> None:
    argilla_aggregation_logic = ComparisonEvaluationAggregationLogic()
    evaluations = (
        ComparisonEvaluation(
            first_player="player_1",
            second_player="player_2" if i < 9000 else "player_3",
            outcome=MatchOutcome.from_rank_literal(
                random.choices([1, 2, 3], [0.5, 0.25, 0.25], k=1)[0]
            ),
        )
        for i in range(10000)
    )
    aggregation = argilla_aggregation_logic.aggregate(evaluations)
    assert aggregation.scores["player_1"].elo > aggregation.scores["player_2"].elo
    assert (
        aggregation.scores["player_3"].elo_standard_error
        > aggregation.scores["player_1"].elo_standard_error
    )
