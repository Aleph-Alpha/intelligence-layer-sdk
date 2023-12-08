from typing import Iterable, Sequence, cast

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
from tests.conftest import in_memory_evaluation_repository  # type: ignore


class StubArgillaClient(ArgillaClient):
    _datasets: dict[str, list[Record]] = Field(default_factory=dict)

    def create_dataset(
        self, workspace_id: str, dataset_name: str, fields: Sequence[Field]
    ) -> str:
        ...

    def add_record(self, dataset_id: str, record: Record) -> None:
        self._datasets.get(dataset_id).append(record)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        ...


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
            return True
        return False

    def aggregate(
        self, evaluations: Iterable[DummyStringEvaluation]
    ) -> DummyStringAggregatedEvaluation:
        evaluations = list(evaluations)
        total = len(evaluations)
        correct_amount = len((b for b in evaluations if b.same == True))
        return DummyStringAggregatedEvaluation(
            percentage_correct=correct_amount / total
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
    stub_argilla_client: StubArgillaClient
) -> DummyStringTaskAgrillaEvaluator:
    return DummyStringTaskAgrillaEvaluator(dummy_string_task, in_memory_evaluation_repository, stub_argilla_client)


def test_argilla_evaluator_can_do_sync_evaluation(
    string_argilla_evaluator: DummyStringTaskAgrillaEvaluator,
) -> None:
    example = Example(DummyStringInput.any(), expected_output=DummyStringOutput.any())
    dataset = SequenceDataset(
        name="dataset",
        examples=[
            example
        ],
    )
    argilla_client = cast(StubArgillaClient, string_argilla_evaluator._client)
    

    overview = string_argilla_evaluator.run_dataset(dataset)
    _ =  string_argilla_evaluator.evaluate_run(dataset, overview)

    for dataset in argilla_client._datasets.values():
        assert any(i == {"input": example.input.input, "completion": example.expected_output.output} for i in dataset)

