from datetime import datetime
from typing import cast
from uuid import uuid4

from _pytest.fixtures import FixtureRequest
from pytest import fixture, mark

from intelligence_layer.core import (
    CompositeTracer,
    InMemoryTaskSpan,
    InMemoryTracer,
    utc_now,
)
from intelligence_layer.evaluation import ExampleTrace, TaskSpanTrace
from intelligence_layer.evaluation.data_storage.run_repository import RunRepository
from intelligence_layer.evaluation.domain import ExampleOutput, RunOverview
from tests.conftest import DummyStringInput

test_repository_fixtures = [
    "file_run_repository",
    "in_memory_run_repository",
]


@fixture
def run_overview() -> RunOverview:
    return RunOverview(
        dataset_id="dataset-id",
        id="run-id-1",
        start=utc_now(),
        end=utc_now(),
        failed_example_count=0,
        successful_example_count=3,
        description="test run overview",
    )


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_example_output_ids_returns_all_sorted_ids(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_ids = [str(uuid4()) for _ in range(10)]
    for run_id in run_ids:
        run_repository.store_example_output(
            ExampleOutput(run_id=run_id, example_id="example_id", output=None)
        )

    example_output_ids = run_repository.example_output_ids()

    assert example_output_ids == sorted(run_ids)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_can_store_and_return_example_evaluation_tracer_and_trace(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_id = "run_id"
    example_id = "example_id"
    now = datetime.now()
    in_memory_tracer = InMemoryTracer()

    tracer = run_repository.example_tracer(run_id, example_id)
    CompositeTracer([tracer, in_memory_tracer]).task_span(
        "task", DummyStringInput(input="input"), now
    )
    example_trace = run_repository.example_trace(run_id, example_id)

    expected_trace = ExampleTrace(
        run_id=run_id,
        example_id=example_id,
        trace=TaskSpanTrace.from_task_span(
            cast(InMemoryTaskSpan, in_memory_tracer.entries[0])
        ),
    )
    assert example_trace == expected_trace


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_outputs_returns_example_outputs_sorted_by_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_id = "run_id"
    example_ids = [str(uuid4()) for _ in range(10)]
    for example_id in example_ids:
        run_repository.store_example_output(
            ExampleOutput(run_id=run_id, example_id=example_id, output=None),
        )

    example_outputs = run_repository.example_outputs(run_id, type(None))

    assert [example_output.example_id for example_output in example_outputs] == sorted(
        example_ids
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_run_repository_stores_and_returns_a_run_overview(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)

    run_repository.store_run_overview(run_overview)
    stored_run_overview = run_repository.run_overview(run_overview.id)

    assert stored_run_overview == run_overview


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_run_overview_returns_non_for_not_existing_run_id(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)

    stored_run_overview = run_repository.run_overview("not-existing-id")

    assert stored_run_overview is None
