from datetime import datetime
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    EvaluationOverview,
    Example,
    ExampleEvaluation,
    FailedExampleEvaluation,
    FileEvaluationRepository,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    RunOverview,
    SequenceDataset,
)
from tests.conftest import DummyStringInput, DummyStringOutput


class DummyStringEvaluation(BaseModel):
    same: bool


class DummyEvaluation(BaseModel):
    result: str


class DummyAggregatedEvaluation(BaseModel):
    score: float


class DummyAggregatedEvaluationWithResultList(BaseModel):
    results: Sequence[DummyEvaluation]


@fixture
def failed_example_result() -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        example_id="other",
        result=FailedExampleEvaluation(error_message="error"),
    )


@fixture
def file_evaluation_repository(tmp_path: Path) -> FileEvaluationRepository:
    return FileEvaluationRepository(tmp_path)


@fixture
def in_memory_evaluation_repository() -> InMemoryEvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def in_memory_dataset_repository_with_dataset(
    dummy_string_dataset: SequenceDataset[DummyStringInput, DummyStringOutput],
) -> InMemoryDatasetRepository:
    dataset_repository = InMemoryDatasetRepository()
    dataset_repository.create_dataset(dummy_string_dataset.name, [(e.input, e.expected_output) for e in dummy_string_dataset.examples])
    return dataset_repository

@fixture
def in_memory_dataset_repository() -> InMemoryDatasetRepository:
    return InMemoryDatasetRepository()


@fixture
def string_dataset_name(
    dummy_string_dataset: SequenceDataset[DummyStringInput, DummyStringOutput],
    in_memory_dataset_repository: InMemoryDatasetRepository
) -> str:
    dataset_name = "name"
    in_memory_dataset_repository.create_dataset(dataset_name, [(e.input, None) for e in dummy_string_dataset.examples])
    return dataset_name


@fixture
def successful_example_result() -> ExampleEvaluation[DummyEvaluation]:
    return ExampleEvaluation(
        example_id="example_id",
        result=DummyEvaluation(result="result"),
    )


@fixture
def dummy_aggregated_evaluation() -> DummyAggregatedEvaluation:
    return DummyAggregatedEvaluation(score=0.5)


@fixture
def evaluation_run_overview(
    dummy_aggregated_evaluation: DummyAggregatedEvaluation,
) -> EvaluationOverview[DummyAggregatedEvaluation]:
    now = datetime.now()
    return EvaluationOverview(
        id="eval-id",
        run_overviews=[
            RunOverview(
                dataset_name="dataset",
                id="run-id",
                start=now,
                end=now,
                failed_example_count=0,
                successful_example_count=0,
            )
        ],
        start=now,
        end=now,
        failed_evaluation_count=3,
        successful_count=5,
        statistics=dummy_aggregated_evaluation,
    )


@fixture
def dummy_string_example() -> Example[DummyStringInput, DummyStringOutput]:
    return Example(
        input=DummyStringInput.any(), expected_output=DummyStringOutput.any()
    )


@fixture
def dummy_string_dataset(
    dummy_string_example: Example[DummyStringInput, DummyStringOutput]
) -> SequenceDataset[DummyStringInput, DummyStringOutput]:
    return SequenceDataset(
        name="dataset",
        examples=[dummy_string_example],
    )
