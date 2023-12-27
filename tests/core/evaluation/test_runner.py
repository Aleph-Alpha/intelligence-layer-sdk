from intelligence_layer.core import (
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    Runner,
)
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

    assert set(output.example_id for output in outputs) == {0, 1, 2}
