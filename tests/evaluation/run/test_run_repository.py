from collections.abc import Iterable, Sequence
from datetime import datetime
from uuid import uuid4

import pytest
from _pytest.fixtures import FixtureRequest
from pytest import fixture, mark

from intelligence_layer.core import CompositeTracer, InMemoryTracer, utc_now
from intelligence_layer.evaluation import ExampleOutput, RunOverview, RunRepository
from intelligence_layer.evaluation.run.domain import FailedExampleRun
from tests.evaluation.conftest import DummyStringInput

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
            labels=set(),
            metadata=dict(),
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
def test_example_output_returns_none_for_not_existing_example_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_id = "run-id"
    example_id = "example-id"
    example_output = ExampleOutput(run_id=run_id, example_id=example_id, output=None)
    run_repository.store_example_output(example_output)

    assert (
        run_repository.example_output(run_id, "not-existing-example-id", type(None))
        is None
    )


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_example_output_returns_none_for_not_existing_run_id(
    repository_fixture: str,
    request: FixtureRequest,
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_id = "run-id"
    example_id = "example-id"
    example_output = ExampleOutput(run_id=run_id, example_id=example_id, output=None)
    run_repository.store_example_output(example_output)

    with pytest.raises(ValueError):
        run_repository.example_output("not-existing-run-id", example_id, type(None))
    with pytest.raises(ValueError):
        run_repository.example_output(
            "not-existing-run-id", "not-existing-example-id", type(None)
        )


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

    tracer = run_repository.create_tracer_for_example(run_id, example_id)
    task_span = CompositeTracer([tracer, in_memory_tracer]).task_span(
        "task", DummyStringInput(input="input"), now
    )
    task_span.end()
    example_tracer = run_repository.example_tracer(run_id, example_id)

    assert tracer
    assert example_tracer
    assert tracer.context == example_tracer.context

    tracer_export = tracer.export_for_viewing()
    example_trace_export = example_tracer.export_for_viewing()

    assert tracer_export == example_trace_export


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

    example_output_ids = run_repository.example_output_ids(run_overview.id)

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

    example_outputs = list(run_repository.example_outputs(run_overview.id, type(None)))

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


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_failed_example_outputs_returns_only_failed_examples(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_repository.store_run_overview(run_overview)

    run_repository.store_example_output(
        ExampleOutput(
            run_id=run_overview.id,
            example_id="1",
            output=FailedExampleRun(error_message="test"),
        )
    )
    run_repository.store_example_output(
        ExampleOutput(run_id=run_overview.id, example_id="2", output=None)
    )

    failed_outputs = list(
        run_repository.failed_example_outputs(
            run_id=run_overview.id, output_type=type(None)
        )
    )

    assert len(failed_outputs) == 1
    assert failed_outputs[0].example_id == "1"


@mark.parametrize(
    "repository_fixture",
    test_repository_fixtures,
)
def test_successful_example_outputs_returns_only_successful_examples(
    repository_fixture: str, request: FixtureRequest, run_overview: RunOverview
) -> None:
    run_repository: RunRepository = request.getfixturevalue(repository_fixture)
    run_repository.store_run_overview(run_overview)

    run_repository.store_example_output(
        ExampleOutput(
            run_id=run_overview.id,
            example_id="1",
            output=FailedExampleRun(error_message="test"),
        )
    )
    run_repository.store_example_output(
        ExampleOutput(run_id=run_overview.id, example_id="2", output=None)
    )

    successful_outputs = list(
        run_repository.successful_example_outputs(
            run_id=run_overview.id, output_type=type(None)
        )
    )

    assert len(successful_outputs) == 1
    assert successful_outputs[0].example_id == "2"
