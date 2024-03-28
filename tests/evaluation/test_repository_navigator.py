from typing import Sequence

import pytest
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer.tracer import TaskSpan
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Dataset, Example
from intelligence_layer.evaluation.dataset.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)
from intelligence_layer.evaluation.evaluation.domain import EvaluationOverview
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.evaluation.evaluator import (
    EvaluationLogic,
    Evaluator,
)
from intelligence_layer.evaluation.evaluation.in_memory_evaluation_repository import (
    InMemoryEvaluationRepository,
)
from intelligence_layer.evaluation.infrastructure.repository_navigator import (
    RepositoryNavigator,
)
from intelligence_layer.evaluation.run.domain import (
    RunOverview,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.run.in_memory_run_repository import (
    InMemoryRunRepository,
)
from intelligence_layer.evaluation.run.run_repository import RunRepository
from intelligence_layer.evaluation.run.runner import Runner


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


@fixture
def dataset_repository() -> DatasetRepository:
    return InMemoryDatasetRepository()


@fixture
def examples() -> Sequence[DummyExample]:
    return [
        DummyExample(input="input0", expected_output="expected_output0", data="data0"),
        DummyExample(input="input1", expected_output="expected_output1", data="data1"),
    ]


@fixture
def dataset(
    dataset_repository: DatasetRepository, examples: Sequence[DummyExample]
) -> Dataset:
    return dataset_repository.create_dataset(
        examples,
        "test",
    )


@fixture
def run_repository() -> RunRepository:
    return InMemoryRunRepository()


@fixture
def run_overview(
    dataset_repository: DatasetRepository,
    run_repository: RunRepository,
    dataset: Dataset,
) -> RunOverview:
    return Runner(
        DummyTask(), dataset_repository, run_repository, "Runner"
    ).run_dataset(dataset.id)


@fixture
def additional_run_overview(
    dataset_repository: DatasetRepository,
    run_repository: RunRepository,
    dataset: Dataset,
) -> RunOverview:
    return Runner(
        DummyTask(), dataset_repository, run_repository, "Runner2"
    ).run_dataset(dataset.id)


@fixture
def eval_repository() -> EvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def eval_overview(
    dataset_repository: DatasetRepository,
    run_repository: RunRepository,
    eval_repository: EvaluationRepository,
    run_overview: RunOverview,
    additional_run_overview: RunOverview,
) -> EvaluationOverview:
    return Evaluator(
        dataset_repository,
        run_repository,
        eval_repository,
        "Evaluator",
        DummyEvalLogic(),
    ).evaluate_runs(run_overview.id, additional_run_overview.id)


@fixture
def repository_navigator(
    dataset_repository: DatasetRepository,
    run_repository: RunRepository,
    eval_repository: EvaluationRepository,
) -> RepositoryNavigator:
    return RepositoryNavigator(dataset_repository, run_repository, eval_repository)


def test_works_on_run_overviews(
    repository_navigator: RepositoryNavigator,
    run_overview: RunOverview,
) -> None:
    # when
    res = list(repository_navigator.run_data(run_overview.id, str, str, str))

    # then
    res = sorted(res, key=lambda result: result.example.input)
    for i in range(2):
        assert res[i].example.input == f"input{i}"
        assert res[i].example.expected_output == f"expected_output{i}"
        assert res[i].output.output == f"input{i} -> output"


def test_works_on_evaluation(
    repository_navigator: RepositoryNavigator,
    eval_overview: EvaluationOverview,
) -> None:
    # when
    res = list(
        repository_navigator.eval_data(eval_overview.id, str, str, str, DummyEval)
    )

    # then
    res = sorted(res, key=lambda result: result.example.input)
    for i in range(2):
        assert res[i].example.input == f"input{i}"
        assert res[i].example.expected_output == f"expected_output{i}"
        assert len(res[i].outputs) == 2
        assert res[i].outputs[0].output == f"input{i} -> output"
        eval_result = res[i].evaluation.result
        assert isinstance(eval_result, DummyEval)
        assert eval_result.eval.startswith(f"input{i}")


def test_initialization_gives_warning_if_not_compatible() -> None:
    dataset_repository = InMemoryDatasetRepository()
    run_repository = InMemoryRunRepository()

    x = RepositoryNavigator(dataset_repository, run_repository)
    with pytest.raises(ValueError):
        list(x.eval_data("irrelevant", str, str, str, DummyEval))


def test_get_run_lineage_for_single_example(
    examples: Sequence[DummyExample],
    repository_navigator: RepositoryNavigator,
    run_overview: RunOverview,
):
    # when
    res = repository_navigator.run_single_example(
        run_overview.id, examples[0].id, str, str, str
    )

    # Then
    assert res.example.input == "input0"
    assert res.output.output == "input0 -> output"


def test_get_eval_lineage_for_single_example(
    examples: Sequence[DummyExample],
    repository_navigator: RepositoryNavigator,
    eval_overview: EvaluationOverview,
):
    # when
    res = repository_navigator.eval_single_example(
        eval_overview.id, examples[0].id, str, str, str, DummyEval
    )

    # Then
    assert res.example.input == "input0"
    assert res.output.output == "input0 -> output"
    assert res.evaluation.result.startswith("input0")
