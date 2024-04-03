from datetime import datetime
from os import getenv
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core import Task, Tracer, utc_now
from intelligence_layer.evaluation import (
    AggregationOverview,
    DatasetRepository,
    EvaluationOverview,
    Example,
    ExampleEvaluation,
    FileAggregationRepository,
    FileEvaluationRepository,
    FileRunRepository,
    InMemoryDatasetRepository,
    InMemoryRunRepository,
    InstructComparisonArgillaAggregationLogic,
    Runner,
    RunOverview,
)
from tests.conftest import DummyStringInput, DummyStringOutput

FAIL_IN_EVAL_INPUT = "fail in eval"
FAIL_IN_TASK_INPUT = "fail in task"


class DummyTask(Task[str, str]):
    def do_run(self, input: str, tracer: Tracer) -> str:
        if input == FAIL_IN_TASK_INPUT:
            raise RuntimeError(input)
        return input


class DummyStringEvaluation(BaseModel):
    same: bool


class DummyEvaluation(BaseModel):
    result: str


class DummyAggregatedEvaluation(BaseModel):
    score: float


class DummyAggregatedEvaluationWithResultList(BaseModel):
    results: Sequence[DummyEvaluation]


@fixture
def sequence_examples() -> Iterable[Example[str, None]]:
    return [
        Example(input="success", expected_output=None, id="example-1"),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None, id="example-2"),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None, id="example-3"),
    ]


@fixture
def evaluation_id() -> str:
    return "evaluation-id-1"


@fixture
def successful_example_evaluation(
    evaluation_id: str,
) -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        evaluation_id=evaluation_id,
        example_id="successful_example",
        result=DummyEvaluation(result="result"),
    )


@fixture
def file_aggregation_repository(tmp_path: Path) -> FileAggregationRepository:
    return FileAggregationRepository(tmp_path)


@fixture
def file_evaluation_repository(tmp_path: Path) -> FileEvaluationRepository:
    return FileEvaluationRepository(tmp_path)


@fixture
def file_run_repository(tmp_path: Path) -> FileRunRepository:
    return FileRunRepository(tmp_path)


@fixture
def string_dataset_id(
    dummy_string_examples: Iterable[Example[DummyStringInput, DummyStringOutput]],
    in_memory_dataset_repository: DatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(
        examples=dummy_string_examples, dataset_name="test-dataset"
    ).id


@fixture
def dummy_aggregated_evaluation() -> DummyAggregatedEvaluation:
    return DummyAggregatedEvaluation(score=0.5)


@fixture
def run_overview() -> RunOverview:
    return RunOverview(
        dataset_id="dataset-id",
        id="run-id-1",
        start=utc_now(),
        end=utc_now(),
        failed_example_count=0,
        successful_example_count=3,
        description="test run overview 1",
    )


@fixture
def evaluation_overview(
    evaluation_id: str, run_overview: RunOverview
) -> EvaluationOverview:
    return EvaluationOverview(
        id=evaluation_id,
        start=utc_now(),
        run_overviews=frozenset([run_overview]),
        description="test evaluation overview 1",
    )


@fixture
def aggregation_overview(
    evaluation_overview: EvaluationOverview,
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
) -> AggregationOverview[DummyAggregatedEvaluation]:
    now = datetime.now()
    return AggregationOverview(
        evaluation_overviews=frozenset([evaluation_overview]),
        id="aggregation-id",
        start=now,
        end=now,
        successful_evaluation_count=5,
        crashed_during_evaluation_count=3,
        description="dummy-evaluator",
        statistics=dummy_aggregated_evaluation,
    )


@fixture
def dummy_string_example() -> Example[DummyStringInput, DummyStringOutput]:
    return Example(
        input=DummyStringInput.any(), expected_output=DummyStringOutput.any()
    )


@fixture
def dummy_string_examples(
    dummy_string_example: Example[DummyStringInput, DummyStringOutput]
) -> Iterable[Example[DummyStringInput, DummyStringOutput]]:
    return [dummy_string_example]


@fixture
def dummy_runner(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> Runner[str, str]:
    return Runner(
        DummyTask(),
        in_memory_dataset_repository,
        in_memory_run_repository,
        "dummy-runner",
    )


@fixture
def argilla_aggregation_logic() -> InstructComparisonArgillaAggregationLogic:
    return InstructComparisonArgillaAggregationLogic()


class StubArgillaClient(ArgillaClient):
    _expected_workspace_id: str
    _expected_fields: Sequence[Field]
    _expected_questions: Sequence[Question]
    _datasets: dict[str, list[RecordData]] = {}
    _score = 3.0

    def ensure_dataset_exists(
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
        dataset_id = str(uuid4())
        self._datasets[dataset_id] = []
        return dataset_id

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        if dataset_id not in self._datasets:
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
def stub_argilla_client() -> StubArgillaClient:
    return StubArgillaClient()


@fixture(scope="session")
def hugging_face_token() -> str:
    load_dotenv()
    token = getenv("HUGGING_FACE_TOKEN")
    assert isinstance(token, str)
    return token
