from typing import Sequence

import pytest
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.evaluation import (
    Dataset,
    DatasetRepository,
    EvaluationLogic,
    EvaluationOverview,
    EvaluationRepository,
    Evaluator,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    RepositoryNavigator,
    Runner,
    RunOverview,
    RunRepository,
    SuccessfulExampleOutput,
    evaluation_lineages_to_pandas,
    run_lineages_to_pandas,
)


class DummyExample(Example[str, str]):
    data: str


class DummyTask(Task[str, str]):
    def do_run(self, input: str, task_span: TaskSpan) -> str:
        return f"{input} -> output"


class DummyEval(BaseModel):
    eval: str


class DummyEvalLogic(EvaluationLogic[str, str, str, DummyEval]):
    def do_evaluate(
        self, example: Example[str, str], *output: SuccessfulExampleOutput[str]
    ) -> DummyEval:
        output_str = ", ".join(o.output for o in output)
        return DummyEval(
            eval=f"{example.input}, {example.expected_output}, {output_str} -> evaluation"
        )


def example_repositories() -> (
    tuple[DatasetRepository, RunRepository, EvaluationRepository]
):
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
        DummyEvalLogic(),
    )
    evaluator.evaluate_runs(run_overview_1, run_overview_2)

    return dataset_repository, run_repository, evaluation_repository
