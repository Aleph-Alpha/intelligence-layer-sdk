from datetime import datetime
from typing import Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core.evaluation.domain import (
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
    RunOverview,
)


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
        run_overviews=[RunOverview(
            dataset_name="dataset",
            id="run-id",
            start=now,
            end=now,
            failed_example_count=0,
            successful_example_count=0,
        )],
        start=now,
        end=now,
        failed_evaluation_count=3,
        successful_count=5,
        statistics=dummy_aggregated_evaluation,
    )
