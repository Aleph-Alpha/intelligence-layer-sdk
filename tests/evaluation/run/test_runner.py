from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import InMemoryTaskSpan, InMemoryTracer
from intelligence_layer.evaluation import (
    Example,
    InMemoryDatasetRepository,
    InMemoryRunRepository,
    Runner,
)
from intelligence_layer.evaluation.run.file_run_repository import FileRunRepository
from tests.evaluation.conftest import (
    FAIL_IN_TASK_INPUT,
    DummyStringExpectedOutput,
    DummyStringInput,
    DummyStringTask,
)


def test_runner_runs_dataset(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    examples = list(sequence_examples)
    task = DummyStringTask()
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

    failed_runs = list(runner.failed_runs(overview.id, Any))  # type: ignore
    assert len(failed_runs) == 1
    assert failed_runs[0].example.id == examples[1].id


def test_runner_works_without_description(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    examples = list(sequence_examples)
    task = DummyStringTask()
    runner = Runner(task, in_memory_dataset_repository, in_memory_run_repository, "")

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name=""
    ).id
    overview = runner.run_dataset(dataset_id)
    assert overview.description is runner.description


def test_runner_has_correct_description(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    examples = list(sequence_examples)
    task = DummyStringTask()
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
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    task = DummyStringTask()
    runner = Runner(
        task, in_memory_dataset_repository, in_memory_run_repository, "dummy-runner"
    )

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=sequence_examples, dataset_name="test-dataset"
    ).id
    with pytest.raises(RuntimeError):
        runner.run_dataset(dataset_id, abort_on_error=True)


def test_runner_resumes_after_error_in_task(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    file_run_repository: FileRunRepository,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    task = DummyStringTask()
    runner = Runner(
        task, in_memory_dataset_repository, file_run_repository, "dummy-runner"
    )

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=sequence_examples, dataset_name="test-dataset"
    ).id

    fail_example_id = ""
    for example in sequence_examples:
        if example.input.input != FAIL_IN_TASK_INPUT:
            continue
        fail_example_id = example.id
    assert fail_example_id != ""

    run_description = "my_run"
    tmp_hash = runner._run_hash(dataset_id, run_description)

    with pytest.raises(RuntimeError):
        runner.run_dataset(dataset_id, abort_on_error=True, description=run_description)

    recovery_data = file_run_repository.finished_examples(tmp_hash)
    assert recovery_data
    assert fail_example_id not in recovery_data.finished_examples

    examples: Sequence[Example[DummyStringInput, DummyStringExpectedOutput]] = (
        in_memory_dataset_repository._datasets_and_examples[dataset_id][1]  # type: ignore
    )
    for example in examples:
        if example.input.input == FAIL_IN_TASK_INPUT:
            example.input.input = "do_not_fail_me"

    runner.run_dataset(
        dataset_id,
        abort_on_error=True,
        description=run_description,
        resume_from_recovery_data=True,
    )
    assert file_run_repository.finished_examples(tmp_hash) is None

    # TODO : we are not yet correctly tracking the number of failed and successful example counts


def test_runner_runs_n_examples(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
) -> None:
    task = DummyStringTask()
    tracer = InMemoryTracer()
    runner = Runner(
        task, in_memory_dataset_repository, in_memory_run_repository, "dummy-runner"
    )
    examples = [
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


def test_runner_run_overview_has_default_metadata_and_labels(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    examples = list(sequence_examples)
    task = DummyStringTask()
    runner = Runner(task, in_memory_dataset_repository, in_memory_run_repository, "foo")

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name=""
    ).id

    overview = runner.run_dataset(dataset_id)

    assert overview.metadata == dict()
    assert overview.labels == set()


def test_runner_run_overview_has_specified_metadata_and_labels(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    run_labels = {"test-label"}
    run_metadata: SerializableDict = dict({"test_key": "test-value"})

    examples = list(sequence_examples)
    task = DummyStringTask()
    runner = Runner(task, in_memory_dataset_repository, in_memory_run_repository, "foo")

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name=""
    ).id
    overview = runner.run_dataset(dataset_id, labels=run_labels, metadata=run_metadata)

    assert overview.metadata == run_metadata
    assert overview.labels == run_labels


def test_run_is_already_computed_works(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    sequence_examples: Iterable[Example[DummyStringInput, DummyStringExpectedOutput]],
) -> None:
    old_model = "old_model"
    examples = list(sequence_examples)
    task = DummyStringTask()
    runner = Runner(task, in_memory_dataset_repository, in_memory_run_repository, "foo")
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=examples, dataset_name=""
    ).id

    run_metadata: SerializableDict = dict({"model": old_model})
    runner.run_dataset(dataset_id, metadata=run_metadata)

    assert runner.run_is_already_computed(dict({"model": old_model}))
    assert not runner.run_is_already_computed(dict({"model": "new_model"}))
