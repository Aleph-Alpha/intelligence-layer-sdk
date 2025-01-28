from collections.abc import Iterable, Sequence
from os import getenv
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from dotenv import load_dotenv
from fsspec.implementations.memory import MemoryFileSystem  # type: ignore
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.connectors.studio.studio import StudioClient
from intelligence_layer.core import Task, TaskSpan, Tracer
from intelligence_layer.evaluation import (
    DatasetRepository,
    Example,
    ExampleEvaluation,
    FileAggregationRepository,
    FileEvaluationRepository,
    FileRunRepository,
)
from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic
from intelligence_layer.evaluation.benchmark.studio_benchmark import (
    StudioBenchmarkRepository,
)
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import (
    SingleOutputEvaluationLogic,
)

FAIL_IN_EVAL_INPUT = "fail in eval"
FAIL_IN_TASK_INPUT = "fail in task"


class DummyTask(Task[str, str]):
    def do_run(self, input: str, tracer: Tracer) -> str:
        if input == FAIL_IN_TASK_INPUT:
            raise RuntimeError(input)
        return input


class DummyEvaluation(BaseModel):
    result: str


class DummyAggregatedEvaluation(BaseModel):
    score: float


class DummyAggregatedEvaluationWithResultList(BaseModel):
    results: Sequence[DummyEvaluation]


class DummyEvaluationLogic(
    SingleOutputEvaluationLogic[
        str,
        str,
        str,
        DummyEvaluation,
    ]
):
    def do_evaluate_single_output(
        self,
        example: Example[str, str],
        output: str,
    ) -> DummyEvaluation:
        if output == FAIL_IN_EVAL_INPUT:
            raise RuntimeError(output)
        return DummyEvaluation(result="Dummy result")


class DummyAggregation(BaseModel):
    num_evaluations: int


class DummyAggregationLogic(AggregationLogic[DummyEvaluation, DummyAggregation]):
    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> DummyAggregation:
        return DummyAggregation(num_evaluations=len(list(evaluations)))


@fixture
def studio_client() -> StudioClient:
    load_dotenv()
    project_name = str(uuid4())
    client = StudioClient(project_name)
    client.create_project(project_name)
    return client


@fixture
def mock_studio_client() -> Mock:
    return Mock(spec=StudioClient)


@fixture
def studio_benchmark_repository(
    mock_studio_client: StudioClient,
) -> StudioBenchmarkRepository:
    return StudioBenchmarkRepository(
        studio_client=mock_studio_client,
    )


@fixture
def studio_dataset_repository(
    mock_studio_client: StudioClient,
) -> StudioDatasetRepository:
    return StudioDatasetRepository(
        studio_client=mock_studio_client,
    )


class DummyStringInput(BaseModel):
    input: str = "dummy-input"


class DummyStringExpectedOutput(BaseModel):
    expected_output: str = "dummy-expected-output"


class DummyStringOutput(BaseModel):
    output: str = "dummy-output"


class DummyStringEvaluation(BaseModel):
    evaluation: str = "dummy-evaluation"


class DummyStringTask(Task[DummyStringInput, DummyStringOutput]):
    def do_run(self, input: DummyStringInput, task_span: TaskSpan) -> DummyStringOutput:
        if input.input == FAIL_IN_TASK_INPUT:
            raise RuntimeError(input)
        return DummyStringOutput()


@fixture
def dummy_string_task() -> DummyStringTask:
    return DummyStringTask()


@fixture
def dummy_string_example() -> Example[DummyStringInput, DummyStringExpectedOutput]:
    return Example(
        input=DummyStringInput(),
        expected_output=DummyStringExpectedOutput(),
        metadata={"some_key": "some_value"},
    )


@fixture
def dummy_string_examples(
    dummy_string_example: Example[DummyStringInput, DummyStringExpectedOutput],
) -> Iterable[Example[DummyStringInput, DummyStringExpectedOutput]]:
    return [dummy_string_example]


@fixture
def dummy_string_dataset_id(
    dummy_string_examples: Iterable[
        Example[DummyStringInput, DummyStringExpectedOutput]
    ],
    in_memory_dataset_repository: DatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(
        examples=dummy_string_examples, dataset_name="test-dataset"
    ).id


@fixture
def sequence_examples() -> (
    Iterable[Example[DummyStringInput, DummyStringExpectedOutput]]
):
    return [
        Example(
            input=DummyStringInput(input="success"),
            expected_output=DummyStringExpectedOutput(),
            id="example-1",
        ),
        Example(
            input=DummyStringInput(input=FAIL_IN_TASK_INPUT),
            expected_output=DummyStringExpectedOutput(),
            id="example-2",
        ),
        Example(
            input=DummyStringInput(input=FAIL_IN_EVAL_INPUT),
            expected_output=DummyStringExpectedOutput(),
            id="example-3",
        ),
    ]


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


@fixture()
def temp_file_system() -> Iterable[MemoryFileSystem]:
    mfs = MemoryFileSystem()

    try:
        yield mfs
    finally:
        mfs.store.clear()


@fixture(scope="session")
def hugging_face_test_repository_id() -> str:
    return f"Aleph-Alpha/IL-temp-tests-{uuid4()}"


@fixture(scope="session")
def hugging_face_token() -> str:
    load_dotenv()
    token = getenv("HUGGING_FACE_TOKEN")
    assert isinstance(token, str)
    return token
