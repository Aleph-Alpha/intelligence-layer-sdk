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
def runner(
    dataset_repository: DatasetRepository, run_repository: RunRepository
) -> Runner[str, str]:
    return Runner(DummyTask(), dataset_repository, run_repository, "Runner")


@fixture
def run_overview(
    runner: Runner[str, str],
    dataset: Dataset,
) -> RunOverview:
    return runner.run_dataset(dataset.id)


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
def evaluation_repository() -> EvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def evaluator(
    dataset_repository: DatasetRepository,
    run_repository: RunRepository,
    evaluation_repository: EvaluationRepository,
) -> Evaluator[str, str, str, DummyEval]:
    return Evaluator(
        dataset_repository,
        run_repository,
        evaluation_repository,
        "Evaluator",
        DummyEvalLogic(),
    )


@fixture
def evaluation_overview(
    evaluator: Evaluator[str, str, str, DummyEval],
    run_overview: RunOverview,
    additional_run_overview: RunOverview,
) -> EvaluationOverview:
    return evaluator.evaluate_runs(run_overview.id, additional_run_overview.id)


@fixture
def repository_navigator(
    dataset_repository: DatasetRepository,
    run_repository: RunRepository,
    evaluation_repository: EvaluationRepository,
) -> RepositoryNavigator:
    return RepositoryNavigator(
        dataset_repository, run_repository, evaluation_repository
    )


def test_works_on_run_overviews(
    repository_navigator: RepositoryNavigator,
    run_overview: RunOverview,
) -> None:
    # when
    res = list(repository_navigator.run_lineages(run_overview.id, str, str, str))

    # then
    res = sorted(res, key=lambda result: result.example.input)
    for i in range(2):
        assert res[i].example.input == f"input{i}"
        assert res[i].example.expected_output == f"expected_output{i}"
        assert res[i].output.output == f"input{i} -> output"


def test_works_run_lineages_work_with_runner(
    runner: Runner[str, str],
    run_overview: RunOverview,
) -> None:
    # when
    res = runner.run_lineages(run_overview.id, str)

    # then
    res = sorted(res, key=lambda result: result.example.input)
    for i in range(2):
        assert res[i].example.input == f"input{i}"
        assert res[i].example.expected_output == f"expected_output{i}"
        assert res[i].output.output == f"input{i} -> output"


def test_works_on_evaluation(
    repository_navigator: RepositoryNavigator,
    evaluation_overview: EvaluationOverview,
) -> None:
    # when
    res = list(
        repository_navigator.evaluation_lineages(
            evaluation_overview.id, str, str, str, DummyEval
        )
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


def test_works_evaluation_lineages_work_with_evaluator(
    evaluator: Evaluator[str, str, str, DummyEval],
    evaluation_overview: EvaluationOverview,
) -> None:
    # when
    res = list(evaluator.evaluation_lineages(evaluation_overview.id))

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
        list(x.evaluation_lineages("irrelevant", str, str, str, DummyEval))
    with pytest.raises(ValueError):
        x.evaluation_lineage("irrelevant", "irrelevant", str, str, str, DummyEval)


def test_get_run_lineage_for_single_example(
    examples: Sequence[DummyExample],
    repository_navigator: RepositoryNavigator,
    run_overview: RunOverview,
) -> None:
    # when
    res = repository_navigator.run_lineage(
        run_overview.id, examples[0].id, str, str, str
    )

    # Then
    assert res is not None
    assert res.example.input == "input0"
    assert res.output.output == "input0 -> output"


def test_get_run_lineage_for_single_example_works_with_runner(
    examples: Sequence[DummyExample],
    runner: Runner[str, str],
    run_overview: RunOverview,
) -> None:
    # when
    res = runner.run_lineage(run_overview.id, examples[0].id, str)

    # Then
    assert res is not None
    assert res.example.input == "input0"
    assert res.output.output == "input0 -> output"


def test_get_eval_lineage_for_single_example(
    examples: Sequence[DummyExample],
    repository_navigator: RepositoryNavigator,
    evaluation_overview: EvaluationOverview,
) -> None:
    # when
    res = repository_navigator.evaluation_lineage(
        evaluation_overview.id, examples[0].id, str, str, str, DummyEval
    )

    # Then
    assert res is not None
    assert res.example.input == "input0"
    assert res.outputs[0].output == "input0 -> output"
    assert len(res.outputs) == 2
    eval_result = res.evaluation.result
    assert isinstance(eval_result, DummyEval)
    assert eval_result.eval.startswith("input0")


def test_get_eval_lineage_for_single_example_works_with_evaluator(
    examples: Sequence[DummyExample],
    evaluator: Evaluator[str, str, str, DummyEval],
    evaluation_overview: EvaluationOverview,
) -> None:
    # when
    res = evaluator.evaluation_lineage(evaluation_overview.id, examples[0].id)

    # Then
    assert res is not None
    assert res.example.input == "input0"
    assert res.outputs[0].output == "input0 -> output"
    assert len(res.outputs) == 2
    eval_result = res.evaluation.result
    assert isinstance(eval_result, DummyEval)
    assert eval_result.eval.startswith("input0")


def test_get_run_lineage_for_non_existent_example_returns_none(
    repository_navigator: RepositoryNavigator,
    run_overview: RunOverview,
) -> None:
    res = repository_navigator.run_lineage(
        run_overview.id, "non-existent-id", str, str, str
    )

    assert res is None


def test_get_eval_lineage_for_non_existent_example_returns_none(
    repository_navigator: RepositoryNavigator,
    evaluation_overview: EvaluationOverview,
) -> None:
    res = repository_navigator.evaluation_lineage(
        evaluation_overview.id, "non-existent-id", str, str, str, DummyEval
    )

    assert res is None


def test_get_run_lineage_for_non_existent_run_id_returns_none(
    repository_navigator: RepositoryNavigator,
) -> None:
    with pytest.raises(ValueError):
        repository_navigator.run_lineage("non-existent-id", "irrelevant", str, str, str)


def test_get_eval_lineage_for_non_existent_eval_id_returns_none(
    repository_navigator: RepositoryNavigator,
) -> None:
    with pytest.raises(ValueError):
        repository_navigator.evaluation_lineage(
            "non-existent-id", "irrelevant", str, str, str, DummyEval
        )


def test_smoke_run_lineage_tree_view(
    repository_navigator: RepositoryNavigator,
    run_overview: RunOverview,
) -> None:
    for lineage in repository_navigator.run_lineages(run_overview.id, str, str, str):
        lineage._rich_render()


def test_smoke_evaluation_lineage_tree_view(
    repository_navigator: RepositoryNavigator,
    evaluation_overview: EvaluationOverview,
) -> None:
    for lineage in repository_navigator.evaluation_lineages(
        evaluation_overview.id, str, str, str, DummyEval
    ):
        lineage._rich_render()


def test_run_lineages_to_pandas(
    repository_navigator: RepositoryNavigator,
    run_overview: RunOverview,
) -> None:
    # Given
    lineages = list(repository_navigator.run_lineages(run_overview.id, str, str, str))
    lineages.sort(key=lambda lineage: (lineage.example.id, lineage.output.run_id))

    # When
    df = run_lineages_to_pandas(lineages).reset_index()
    # df.sort_index(inplace=True)

    # Then
    assert [lineage.example.id for lineage in lineages] == df["example_id"].to_list()
    assert [lineage.output.run_id for lineage in lineages] == df["run_id"].to_list()
    assert [lineage.example.input for lineage in lineages] == df["input"].to_list()
    assert [lineage.example.expected_output for lineage in lineages] == df[
        "expected_output"
    ].to_list()
    assert [lineage.output.output for lineage in lineages] == df["output"].to_list()
    assert lineages == df["lineage"].to_list()


def test_evaluation_lineages_to_pandas(
    repository_navigator: RepositoryNavigator,
    evaluation_overview: EvaluationOverview,
) -> None:
    # Given
    lineages = list(
        repository_navigator.evaluation_lineages(
            evaluation_overview.id, str, str, str, DummyEval
        )
    )

    # When
    df = evaluation_lineages_to_pandas(lineages)

    # Then
    count = 0
    for lineage in lineages:
        for output in lineage.outputs:
            row = df.loc[
                lineage.example.id, lineage.evaluation.evaluation_id, output.run_id  # type: ignore
            ]
            assert lineage.example.input == row.input
            assert lineage.example.expected_output == row.expected_output
            assert output.output == row.output
            assert lineage == row.lineage
            count += 1

    assert count == len(df)
