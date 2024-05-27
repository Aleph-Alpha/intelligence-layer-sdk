from typing import Iterable

import pytest

from intelligence_layer.core import InMemoryTaskSpan, InMemoryTracer
from intelligence_layer.evaluation import (
    Example,
    InMemoryDatasetRepository,
    InMemoryRunRepository,
    Runner,
)
from tests.evaluation.conftest import FAIL_IN_TASK_INPUT, DummyTask


def test_runner_runs_dataset(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    examples = list(sequence_examples)
    task = DummyTask()
    runner = Runner(
        task, in_memory_dataset_repository, in_memory_run_repository, "dummy-runner"
    )

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name="test-dataset"
    ).id
    overview = runner.run_dataset(dataset_id)
    outputs = list(
        in_memory_run_repository.example_outputs(
            overview.id, output_type=runner.output_type()
        )
    )

    assert set(output.example_id for output in outputs) == set(
        example.id for example in examples
    )

    failed_runs = list(runner.failed_runs(overview.id, type(None)))
    assert len(failed_runs) == 1
    assert failed_runs[0].example.id == examples[1].id


def test_runner_works_without_description(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    examples = list(sequence_examples)
    task = DummyTask()
    runner = Runner(task, in_memory_dataset_repository, in_memory_run_repository, "")

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name=""
    ).id
    overview = runner.run_dataset(dataset_id)
    assert overview.description is runner.description


def test_runner_has_correct_description(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    examples = list(sequence_examples)
    task = DummyTask()
    runner = Runner(task, in_memory_dataset_repository, in_memory_run_repository, "foo")

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name=""
    ).id
    run_description = "bar"
    overview = runner.run_dataset(dataset_id, description=run_description)

    assert runner.description in overview.description
    assert run_description in overview.description


def test_runner_aborts_on_error(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[str, None]],
) -> None:
    task = DummyTask()
    runner = Runner(
        task, in_memory_dataset_repository, in_memory_run_repository, "dummy-runner"
    )

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=sequence_examples, dataset_name="test-dataset"
    ).id
    with pytest.raises(RuntimeError):
        runner.run_dataset(dataset_id, abort_on_error=True)


def test_runner_runs_n_examples(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> None:
    task = DummyTask()
    tracer = InMemoryTracer()
    runner = Runner(
        task, in_memory_dataset_repository, in_memory_run_repository, "dummy-runner"
    )
    examples = [
        Example(input="success", expected_output=None, id="example-1"),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None, id="example-2"),
    ]

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name="test-dataset"
    ).id
    overview = runner.run_dataset(dataset_id)
    overview_with_tracer = runner.run_dataset(dataset_id, tracer, num_examples=1)

    assert overview.failed_example_count == 1
    assert overview.successful_example_count == 1
    assert overview_with_tracer.successful_example_count == 1
    assert overview_with_tracer.failed_example_count == 0

    entries = tracer.entries
    assert len(entries) == 1
    assert all([isinstance(e, InMemoryTaskSpan) for e in entries])
