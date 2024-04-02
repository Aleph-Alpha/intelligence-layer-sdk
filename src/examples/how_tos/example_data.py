from typing import Sequence

from pydantic import BaseModel

from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.evaluation import (
    Dataset,
    EvaluationLogic,
    EvaluationOverview,
    Evaluator,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    Runner,
    RunOverview,
    SuccessfulExampleOutput,
)


class DummyExample(Example[str, str]):
    data: str


class DummyTask(Task[str, str]):
    def do_run(self, input: str, task_span: TaskSpan) -> str:
        return f"{input} -> output"


class DummyEvaluation(BaseModel):
    eval: str


class DummyEvaluationLogic(EvaluationLogic[str, str, str, DummyEvaluation]):
    def do_evaluate(
        self, example: Example[str, str], *output: SuccessfulExampleOutput[str]
    ) -> DummyEvaluation:
        output_str = "(" + (", ".join(o.output for o in output)) + ")"
        return DummyEvaluation(
            eval=f"{example.input}, {example.expected_output}, {output_str} -> evaluation"
        )


class ExampleData:
    examples: Sequence[DummyExample]
    dataset_repository: InMemoryDatasetRepository
    run_repository: InMemoryRunRepository
    evaluation_repository: InMemoryEvaluationRepository
    runner: Runner[str, str]
    evaluator: Evaluator[str, str, str, DummyEvaluation]
    dataset: Dataset
    run_overview_1: RunOverview
    run_overview_2: RunOverview
    evaluation_overview: EvaluationOverview


def example_data() -> ExampleData:
    examples = [
        DummyExample(input="input0", expected_output="expected_output0", data="data0"),
        DummyExample(input="input1", expected_output="expected_output1", data="data1"),
    ]

    dataset_repository = InMemoryDatasetRepository()
    dataset = dataset_repository.create_dataset(
        examples=examples, dataset_name="my-dataset"
    )

    run_repository = InMemoryRunRepository()
    runner = Runner(DummyTask(), dataset_repository, run_repository, "my-runner")
    run_overview_1 = runner.run_dataset(dataset.id)
    run_overview_2 = runner.run_dataset(dataset.id)

    evaluation_repository = InMemoryEvaluationRepository()
    evaluator = Evaluator(
        dataset_repository,
        run_repository,
        evaluation_repository,
        "my-evaluator",
        DummyEvaluationLogic(),
    )
    evaluation_overview = evaluator.evaluate_runs(run_overview_1.id, run_overview_2.id)

    example_data = ExampleData()
    example_data.examples = examples
    example_data.dataset_repository = dataset_repository
    example_data.run_repository = run_repository
    example_data.evaluation_repository = evaluation_repository
    example_data.runner = runner
    example_data.evaluator = evaluator
    example_data.dataset = dataset
    example_data.run_overview_1 = run_overview_1
    example_data.run_overview_2 = run_overview_2
    example_data.evaluation_overview = evaluation_overview
    return example_data
