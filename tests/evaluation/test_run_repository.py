from datetime import datetime
from itertools import product
from typing import Iterable, Sequence, cast
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
def run_overviews() -> Sequence[RunOverview]:
    run_overview_ids = [str(uuid4()) for _ in range(10)]
    run_overviews = []
    for run_id in run_overview_ids:
        run_overview = RunOverview(
            dataset_id="dataset-id",
            id=run_id,
            start=utc_now(),
            end=utc_now(),
            failed_example_count=0,
            successful_example_count=1,
            description="test run overview",
        )
        run_overviews.append(run_overview)
    return run_overviews


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_run_repository_stores_and_returns_example_output(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_id = "run-id"
    example_id = "example-id"
    example_output = ExampleOutput(run_id=run_id, example_id=example_id, output=None)

    run_repository.store_example_output(example_output)
    stored_example_output = run_repository.example_output(
        run_id, example_id, type(None)
    )

    assert stored_example_output == example_output


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_output_returns_none_for_not_existing_ids(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_id = "run-id"
    example_id = "example-id"
    example_output = ExampleOutput(run_id=run_id, example_id=example_id, output=None)
    run_repository.store_example_output(example_output)

    stored_example_outputs = [
        run_repository.example_output("not-existing-run-id", example_id, type(None)),
        run_repository.example_output(run_id, "not-existing-example-id", type(None)),
        run_repository.example_output(
            "not-existing-run-id", "not-existing-example-id", type(None)
        ),
    ]

    assert stored_example_outputs == [None, None, None]


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
def test_example_outputs_returns_sorted_example_outputs(
    repository_fixture: str,
    request: FixtureRequest,
    run_overviews: Sequence[RunOverview],
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    some_run_overviews = run_overviews[:2]

    for run_overview in some_run_overviews:
        run_repository.store_run_overview(run_overview)

    example_ids = [str(uuid4()) for _ in range(10)]
    expected_example_outputs = []
    for run_id, example_id in product(
        [run_overview.id for run_overview in some_run_overviews], example_ids
    ):
        example_output = ExampleOutput(
            run_id=run_id, example_id=example_id, output=None
        )
        run_repository.store_example_output(example_output)
        expected_example_outputs.append(example_output)

    example_outputs = list(run_repository.example_outputs(type(None)))

    assert example_outputs == sorted(
        expected_example_outputs,
        key=lambda example: (example.run_id, example.example_id),
    )


@mark.parametrize("repository_fixture", test_repository_fixtures)
def test_run_example_output_ids_returns_all_sorted_ids(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_repository.store_run_overview(run_overview)
    example_ids = [str(uuid4()) for _ in range(10)]
    for example_id in example_ids:
        example_output = ExampleOutput(
            run_id=run_overview.id, example_id=example_id, output=None
        )
        run_repository.store_example_output(example_output)

    example_output_ids = run_repository.run_example_output_ids(run_overview.id)

    assert example_output_ids == sorted(example_ids)


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_run_example_outputs_returns_sorted_run_example_outputs(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_repository.store_run_overview(run_overview)
    example_ids = [str(uuid4()) for _ in range(10)]
    expected_example_outputs = []
    for example_id in example_ids:
        example_output = ExampleOutput(
            run_id=run_overview.id, example_id=example_id, output=None
        )
        run_repository.store_example_output(example_output)
        expected_example_outputs.append(example_output)

    example_outputs = list(
        run_repository.run_example_outputs(run_overview.id, type(None))
    )

    assert example_outputs == sorted(
        expected_example_outputs,
        key=lambda example: (example.run_id, example.example_id),
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
def test_run_overview_returns_none_for_not_existing_run_id(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)

    stored_run_overview = run_repository.run_overview("not-existing-id")

    assert stored_run_overview is None


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_run_overviews_returns_all_sorted_run_overviews(
    repository_fixture: str,
    request: FixtureRequest,
    run_overviews: Iterable[RunOverview],
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)

    for run_overview in run_overviews:
        run_repository.store_run_overview(run_overview)

    stored_run_overviews = list(run_repository.run_overviews())

    assert stored_run_overviews == sorted(
        run_overviews, key=lambda overview: overview.id
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_run_overview_ids_returns_all_sorted_ids(
    repository_fixture: str,
    request: FixtureRequest,
    run_overviews: Iterable[RunOverview],
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_overview_ids = [run_overview.id for run_overview in run_overviews]
    for run_overview in run_overviews:
        run_repository.store_run_overview(run_overview)

    stored_run_overview_ids = list(run_repository.run_overview_ids())

    assert stored_run_overview_ids == sorted(run_overview_ids)
