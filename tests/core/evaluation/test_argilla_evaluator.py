from typing import Iterable, Sequence

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
    def create_dataset(
        self, workspace_id: str, dataset_name: str, fields: Sequence[Field]
    ) -> str:
        ...

    def add_record(self, dataset_id: str, record: Record) -> None:
        ...

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
def dummy_string_task() -> DummyStringTask:
    return DummyStringTask()


@fixture
def argilla_evaluator(
    dummy_string_task: DummyStringTask,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
) -> ArgillaEvaluator:
    client = StubArgillaClient()
    return ArgillaEvaluator(dummy_string_task, in_memory_evaluation_repository, client)


def test_argilla_evaluator_can_do_sync_evaluation(
    argilla_evaluator: ArgillaEvaluator,
) -> None:
    dataset = SequenceDataset(
        name="dataset",
        examples=[
            Example(DummyStringInput.any(), expected_output=DummyStringOutput.any())
        ],
    )

    overview = argilla_evaluator.run_dataset(dataset)
    partial_overview = argilla_evaluator.evaluate_run(dataset, overview)
    full_overview = argilla_evaluator.aggregate_evaluation(partial_overview.id)
