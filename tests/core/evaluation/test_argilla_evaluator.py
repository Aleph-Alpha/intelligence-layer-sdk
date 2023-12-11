from typing import Iterable, Sequence, cast
from uuid import uuid4

from faker import Faker
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core import (
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    Example,
    InMemoryEvaluationRepository,
    SequenceDataset,
    Task,
    TaskSpan,
)
from tests.conftest import in_memory_evaluation_repository  # noqa: W0611


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
            )
            for _ in dataset
        ]


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
    average_human_eval_score: float


class DummyStringTaskAgrillaEvaluator(
    ArgillaEvaluator[
        DummyStringInput,
        DummyStringOutput,
        DummyStringOutput,
        DummyStringAggregatedEvaluation,
    ]
):
    def aggregate(
        self,
        evaluations: Iterable[ArgillaEvaluation],
    ) -> DummyStringAggregatedEvaluation:
        evaluations = list(evaluations)
        total_human_score = sum(
            cast(float, a.responses["human-score"]) for a in evaluations
        )
        return DummyStringAggregatedEvaluation(
            average_human_eval_score=total_human_score / len(evaluations),
        )

    def _to_record(
        self, example_id: str, input: DummyStringInput, output: DummyStringOutput
    ) -> RecordData:
        return RecordData(
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
    in_memory_evaluation_repository: InMemoryEvaluationRepository,  # noqa: w0404
    stub_argilla_client: StubArgillaClient,
) -> DummyStringTaskAgrillaEvaluator:
    stub_argilla_client._expected_workspace_id = "workspace-id"
    questions = [
        Question(
            name="question",
            title="title",
            description="description",
            options=[1],
        )
    ]
    fields = [
        Field(name="output", title="Output"),
        Field(name="input", title="Input"),
    ]
    evaluator = DummyStringTaskAgrillaEvaluator(
        dummy_string_task,
        ArgillaEvaluationRepository(
            in_memory_evaluation_repository, stub_argilla_client
        ),
        stub_argilla_client,
        stub_argilla_client._expected_workspace_id,
        fields,
        questions,
    )
    stub_argilla_client._expected_questions = questions
    stub_argilla_client._expected_fields = fields
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

    overview = string_argilla_evaluator.partial_evaluate_dataset(dataset)

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
    eval_overview = string_argilla_evaluator.partial_evaluate_dataset(dataset)
    aggregated_eval_overview = string_argilla_evaluator.aggregate_evaluation(
        eval_overview.id
    )

    assert (
        aggregated_eval_overview.statistics.average_human_eval_score
        == argilla_client._score
    )
