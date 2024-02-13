from intelligence_layer.core import (
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    Runner,
)
from intelligence_layer.core.tracer import InMemoryTracer
from tests.core.evaluation.conftest import (
    FAIL_IN_EVAL_INPUT,
    FAIL_IN_TASK_INPUT,
    DummyTask,
)


def test_runner_runs_dataset(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> None:
    task = DummyTask()
    runner = Runner(
        task,
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "dummy-runner",
    )
    examples = [
        Example(input="success", expected_output=None),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None),
    ]

    dataset_id = in_memory_dataset_repository.create_dataset(examples=examples)
    overview = runner.run_dataset(dataset_id)
    outputs = list(
        in_memory_evaluation_repository.example_outputs(
            overview.id, output_type=runner.output_type()
        )
    )

    assert set(output.example_id for output in outputs) == set(
        example.id for example in examples
    )


def test_runner_runs_n_examples(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> None:
    task = DummyTask()
    tracer = InMemoryTracer()
    runner = Runner(
        task,
        in_memory_evaluation_repository,
        in_memory_dataset_repository,
        "dummy-runner",
    )
    examples = [
        Example(input="success", expected_output=None),
        Example(input=FAIL_IN_TASK_INPUT, expected_output=None),
        Example(input=FAIL_IN_EVAL_INPUT, expected_output=None),
    ]

    dataset_id = in_memory_dataset_repository.create_dataset(examples=examples)
    overview = runner.run_dataset(dataset_id, num_examples=2)
    overview_with_tracer = runner.run_dataset(dataset_id, tracer, 1)

    assert overview.failed_example_count == 1
    assert overview.successful_example_count == 1
    assert overview_with_tracer.successful_example_count == 1
    assert overview_with_tracer.failed_example_count == 0
