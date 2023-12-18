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
from intelligence_layer.core import (
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
)
from intelligence_layer.core.evaluation.runner import Runner
from tests.conftest import DummyStringInput, DummyStringOutput, DummyStringTask
from tests.core.evaluation.conftest import DummyAggregatedEvaluation


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


class DummyStringTaskArgillaEvaluator(
    ArgillaEvaluator[
        DummyStringInput,
        DummyStringOutput,
        DummyStringOutput,
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

    # mypy expects *args where this method only uses one output
    def _to_record(  # type: ignore
        self,
        example: Example[DummyStringInput, DummyStringOutput],
        output: DummyStringOutput,
    ) -> Sequence[RecordData]:
        return [
            RecordData(
                content={
                    "input": example.input.input,
                    "output": output.output,
                },
                example_id=example.id,
            )
        ]


@fixture
def stub_argilla_client() -> StubArgillaClient:
    return StubArgillaClient()


@fixture
def string_argilla_evaluator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,  # noqa: w0404
    in_memory_dataset_repository: InMemoryDatasetRepository,
    stub_argilla_client: StubArgillaClient,
) -> DummyStringTaskArgillaEvaluator:
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
    evaluator = DummyStringTaskArgillaEvaluator(
        ArgillaEvaluationRepository(
            in_memory_evaluation_repository, stub_argilla_client
        ),
        in_memory_dataset_repository,
        stub_argilla_client._expected_workspace_id,
        fields,
        questions,
    )
    stub_argilla_client._expected_questions = questions
    stub_argilla_client._expected_fields = fields
    return evaluator


@fixture
def string_argilla_runner(
    dummy_string_task: DummyStringTask,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,  # noqa: w0404
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> Runner[DummyStringInput, DummyStringOutput]:
    return Runner(
        dummy_string_task,
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "dummy-task",
    )


def test_argilla_evaluator_can_do_sync_evaluation(
    string_argilla_evaluator: DummyStringTaskArgillaEvaluator,
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
) -> None:
    argilla_client = cast(StubArgillaClient, string_argilla_evaluator._client)

    run_overview = string_argilla_runner.run_dataset(string_dataset_id)
    eval_overview = string_argilla_evaluator.partial_evaluate_dataset(run_overview.id)
    dummy_string_dataset = string_argilla_evaluator._dataset_repository.dataset(
        string_dataset_id, DummyStringInput, DummyStringOutput
    )
    assert dummy_string_dataset is not None

    assert eval_overview.id in argilla_client._datasets
    saved_dataset = argilla_client._datasets[eval_overview.id]
    examples = list(dummy_string_dataset)
    assert len(saved_dataset) == len(examples)
    assert saved_dataset[0].example_id == examples[0].id
    assert saved_dataset[0].content["input"] == examples[0].input.input


def test_argilla_evaluator_can_aggregate_evaluation(
    string_argilla_evaluator: DummyStringTaskArgillaEvaluator,
    string_argilla_runner: Runner[DummyStringInput, DummyStringOutput],
    string_dataset_id: str,
) -> None:
    argilla_client = cast(StubArgillaClient, string_argilla_evaluator._client)
    run_overview = string_argilla_runner.run_dataset(string_dataset_id)
    eval_overview = string_argilla_evaluator.partial_evaluate_dataset(run_overview.id)
    aggregated_eval_overview = string_argilla_evaluator.aggregate_evaluation(
        eval_overview.id
    )

    assert aggregated_eval_overview.statistics.score == argilla_client._score
