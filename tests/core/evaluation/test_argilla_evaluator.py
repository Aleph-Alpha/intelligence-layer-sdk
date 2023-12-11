from collections import defaultdict
from random import randint
from typing import Iterable, Sequence, cast
from uuid import uuid4

from faker import Faker
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Record,
)
from intelligence_layer.core import (
    ArgillaEvaluator,
    Example,
    InMemoryEvaluationRepository,
    SequenceDataset,
    Task,
    TaskSpan,
)
from tests.conftest import in_memory_evaluation_repository


class StubArgillaClient(ArgillaClient):
    _expected_workspace_id: str
    _expected_fields: Sequence[Field] = []
    _datasets: dict[str, list[Record]] = {}
    _dataset_name = ""
    _score = 3.0

    def create_dataset(
        self, workspace_id: str, dataset_name: str, fields: Sequence[Field]
    ) -> str:
        if workspace_id != self._expected_workspace_id:
            raise Exception("Incorrect workspace id")
        if fields != self._expected_fields:
            raise Exception("Incorrect fields")
        self._datasets[dataset_name] = []
        return dataset_name

    def add_record(self, dataset_id: str, record: Record) -> None:
        if dataset_id not in self._datasets:
            raise Exception("Add record: dataset not found") 
        self._datasets[dataset_id].append(record)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        dataset = self._datasets.get(dataset_id)
        assert dataset
        return [{"human-score": self._score} for _ in dataset]


class DummyStringInput(BaseModel):
    input: str

    @classmethod
    def any(cls) -> "DummyStringInput":
        fake = Faker()
        return cls(input=fake.text())


class DummyStringOutput(BaseModel):
    output: str

    @classmethod
    def any(cls) -> "DummyStringOutput":
        fake = Faker()
        return cls(output=fake.text())


class DummyStringTask(Task[DummyStringInput, DummyStringOutput]):
    def do_run(self, input: DummyStringInput, task_span: TaskSpan) -> DummyStringOutput:
        return DummyStringOutput.any()


class DummyStringEvaluation(BaseModel):
    same: bool


class DummyStringAggregatedEvaluation(BaseModel):
    percentage_correct: float
    human_eval_score: float


class DummyStringTaskAgrillaEvaluator(
    ArgillaEvaluator[
        DummyStringInput,
        DummyStringOutput,
        DummyStringOutput,
        DummyStringEvaluation,
        DummyStringAggregatedEvaluation,
    ]
):
    def do_evaluate(
        self,
        input: DummyStringInput,
        _: DummyStringOutput,
        expected_output: DummyStringOutput,
    ) -> DummyStringEvaluation:
        if input.input == expected_output.output:
            return DummyStringEvaluation(same=True)
        return DummyStringEvaluation(same=False)

    def aggregate(
        self, evaluations: Iterable[DummyStringEvaluation]
    ) -> DummyStringAggregatedEvaluation:
        evaluations = list(evaluations)
        total = len(evaluations)
        correct_amount = len([b for b in evaluations if b.same == True])
        return DummyStringAggregatedEvaluation(
            percentage_correct=correct_amount / total, human_eval_score=0
        )

    def _dataset_info(self) -> tuple[str, Sequence[Field]]:
        return [
            "Dummy string evaluator dataset name",
            [
                Field(name="input", title="Input"),
                Field(name="output", title="Output"),
            ],
        ]

    def _to_record(
        self, example_id: str, input: DummyStringInput, output: DummyStringOutput
    ) -> Record:
        return Record(
            content={
                "input": input.input,
                "output": output.output,
            },
            example_id=example_id,
        )


@fixture
def stub_argilla_client() -> StubArgillaClient:
    return StubArgillaClient()


@fixture
def dummy_string_task() -> DummyStringTask:
    return DummyStringTask()


@fixture
def string_argilla_evaluator(
    dummy_string_task: DummyStringTask,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    stub_argilla_client: StubArgillaClient,
) -> DummyStringTaskAgrillaEvaluator:
    stub_argilla_client._expected_workspace_id = "workspace-id"
    evaluator = DummyStringTaskAgrillaEvaluator(
        dummy_string_task,
        in_memory_evaluation_repository,
        stub_argilla_client,
        stub_argilla_client._expected_workspace_id,
    )
    stub_argilla_client._expected_fields = evaluator._dataset_info()
    return evaluator


def test_argilla_evaluator_can_do_sync_evaluation(
    string_argilla_evaluator: DummyStringTaskAgrillaEvaluator,
) -> None:
    example = Example(
        input=DummyStringInput.any(), expected_output=DummyStringOutput.any()
    )
    dataset = SequenceDataset(
        name="dataset",
        examples=[example],
    )
    argilla_client = cast(StubArgillaClient, string_argilla_evaluator._client)
    run_overview = string_argilla_evaluator.run_dataset(dataset)
    overview = string_argilla_evaluator.evaluate_run(dataset, run_overview)
    assert overview.id in argilla_client._datasets
    saved_dataset = argilla_client._datasets[overview.id]
    assert len(saved_dataset) == len(dataset.examples)
    assert saved_dataset[0].example_id == example.id
    assert saved_dataset[0].content["input"] == example.input.input


def test_argilla_evaluator_can_aggregate_evaluation(
    string_argilla_evaluator: DummyStringTaskAgrillaEvaluator,
) -> None:
    example = Example(
        input=DummyStringInput.any(), expected_output=DummyStringOutput.any()
    )
    dataset = SequenceDataset(
        name="dataset",
        examples=[example],
    )
    argilla_client = cast(StubArgillaClient, string_argilla_evaluator._client)
    run_overview = string_argilla_evaluator.run_dataset(dataset)
    eval_overview = string_argilla_evaluator.evaluate_run(dataset, run_overview)
    aggregated_eval_overview = string_argilla_evaluator.aggregate_evaluation(
        eval_overview.id
    )

    assert aggregated_eval_overview.statistics.human_eval_score == argilla_client._score
    assert aggregated_eval_overview.statistics.percentage_correct == 1.0
