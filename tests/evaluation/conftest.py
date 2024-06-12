from collections.abc import Iterable, Sequence
from os import getenv
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fsspec.implementations.memory import MemoryFileSystem  # type: ignore
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import Task, TaskSpan, Tracer
from intelligence_layer.evaluation import (
    DatasetRepository,
    Example,
    ExampleEvaluation,
    FileAggregationRepository,
    FileEvaluationRepository,
    FileRunRepository,
    InMemoryDatasetRepository,
    InMemoryRunRepository,
    Runner,
)

FAIL_IN_EVAL_INPUT = "fail in eval"
FAIL_IN_TASK_INPUT = "fail in task"


class DummyStringInput(BaseModel):
    input: str = "dummy-input"


class DummyStringOutput(BaseModel):
    output: str = "dummy-output"


class DummyStringEvaluation(BaseModel):
    evaluation: str = "dummy-evaluation"


class DummyStringTask(Task[DummyStringInput, DummyStringOutput]):
    def do_run(self, input: DummyStringInput, task_span: TaskSpan) -> DummyStringOutput:
        return DummyStringOutput()


@fixture
def dummy_string_task() -> DummyStringTask:
    return DummyStringTask()


@fixture
def string_dataset_id(
    dummy_string_examples: Iterable[Example[DummyStringInput, DummyStringOutput]],
    in_memory_dataset_repository: DatasetRepository,
) -> str:
    return in_memory_dataset_repository.create_dataset(
        examples=dummy_string_examples, dataset_name="test-dataset"
    ).id


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


@fixture
def sequence_examples() -> Iterable[Example[str, None]]:
    return [
        Example(input="success", expected_output=None, id="example-1"),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None, id="example-2"),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None, id="example-3"),
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


@fixture
def dummy_string_example() -> Example[DummyStringInput, DummyStringOutput]:
    return Example(
        input=DummyStringInput(),
        expected_output=DummyStringOutput(),
        metadata={"some_key": "some_value"},
    )


@fixture
def dummy_string_examples(
    dummy_string_example: Example[DummyStringInput, DummyStringOutput],
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


@fixture()
def temp_file_system() -> Iterable[MemoryFileSystem]:
    mfs = MemoryFileSystem()

    try:
        yield mfs
    finally:
        mfs.store.clear()


@fixture(scope="session")
def hugging_face_test_repository_id() -> str:
    return f"Aleph-Alpha/IL-temp-tests-{str(uuid4())}"


@fixture(scope="session")
def hugging_face_token() -> str:
    load_dotenv()
    token = getenv("HUGGING_FACE_TOKEN")
    assert isinstance(token, str)
    return token
