from collections.abc import Sequence
from typing import TypeVar

import pytest
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.core.tracer.tracer import utc_now
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
from intelligence_layer.evaluation.aggregation.domain import AggregationOverview
from intelligence_layer.evaluation.infrastructure.repository_navigator import (
    aggregation_overviews_to_pandas,
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


class AggregationDummy(BaseModel):
    score: float = 0.5
    value: float = 0.3


T = TypeVar("T", bound=BaseModel)


def create_aggregation_overview(
    statistics: T,
) -> AggregationOverview[T]:
    return AggregationOverview(
        evaluation_overviews=frozenset(),
        id="aggregation-id",
        start=utc_now(),
        end=utc_now(),
        successful_evaluation_count=5,
        crashed_during_evaluation_count=3,
        description="dummy-evaluator",
        statistics=statistics,
    )


@pytest.mark.parametrize("length", [1, 2])
def test_aggregation_overviews_to_pandas(length: int) -> None:
    # given
    overview = create_aggregation_overview(AggregationDummy())
    # when
    df = aggregation_overviews_to_pandas(
        [overview] * length, unwrap_statistics=False, unwrap_metadata=False
    )
    # then
    assert len(df) == length
    assert set(AggregationOverview.model_fields.keys()) == set(df.columns)


def test_aggregation_overviews_to_pandas_unwrap_statistics() -> None:
    overview = create_aggregation_overview(AggregationDummy())

    df = aggregation_overviews_to_pandas([overview], unwrap_statistics=True)

    assert "score" in df.columns
    assert "value" in df.columns
    assert "statistics" not in df.columns
    assert all(df["score"] == 0.5)
    assert all(df["value"] == 0.3)

    class AggregationDummy2(BaseModel):
        score_2: float = 0.5
        value_2: float = 0.3

    overview2 = create_aggregation_overview(AggregationDummy2())

    df = aggregation_overviews_to_pandas([overview2], unwrap_statistics=True)
    assert "score_2" in df.columns
    assert "value_2" in df.columns
    assert "statistics" not in df.columns


def test_aggregation_overviews_to_pandas_unwrap_metadata() -> None:
    # given

    overview = AggregationOverview(
        evaluation_overviews=frozenset([]),
        id="aggregation-id",
        start=utc_now(),
        end=utc_now(),
        successful_evaluation_count=5,
        crashed_during_evaluation_count=3,
        description="dummy-evaluator",
        statistics=AggregationDummy(),
        labels=set(),
        metadata=dict({"model": "model_a", "prompt": "prompt_a"}),
    )
    overview2 = AggregationOverview(
        evaluation_overviews=frozenset([]),
        id="aggregation-id2",
        start=utc_now(),
        end=utc_now(),
        successful_evaluation_count=5,
        crashed_during_evaluation_count=3,
        description="dummy-evaluator",
        statistics=AggregationDummy(),
        labels=set(),
        metadata=dict(
            {"model": "model_a", "prompt": "prompt_a", "different_column": "value"}
        ),
    )

    df = aggregation_overviews_to_pandas(
        [overview, overview2], unwrap_metadata=True, strict=False
    )

    assert "model" in df.columns
    assert "prompt" in df.columns
    assert "different_column" in df.columns
    assert "metadata" not in df.columns
    assert all(df["model"] == "model_a")
    assert all(df["prompt"] == "prompt_a")


def test_aggregation_overviews_to_pandas_works_with_eval_overviews() -> None:
    # given
    eval_overview = EvaluationOverview(
        run_overviews=frozenset(),
        id="id",
        start_date=utc_now(),
        end_date=utc_now(),
        successful_evaluation_count=1,
        failed_evaluation_count=1,
        description="",
        labels=set(),
        metadata=dict(),
    )
    overview = AggregationOverview(
        evaluation_overviews=frozenset([eval_overview]),
        id="aggregation-id",
        start=utc_now(),
        end=utc_now(),
        successful_evaluation_count=5,
        crashed_during_evaluation_count=3,
        description="dummy-evaluator",
        statistics=AggregationDummy(),
    )
    # when
    df = aggregation_overviews_to_pandas([overview], unwrap_statistics=False)
    # then
    assert len(df) == 1


def test_aggregation_overviews_to_pandas_works_with_empty_input() -> None:
    # when
    df = aggregation_overviews_to_pandas([])
    # then
    assert len(df) == 0


def test_aggregation_overviews_does_not_work_with_different_aggregations() -> None:
    # given
    overview = create_aggregation_overview(AggregationDummy())

    class OtherVariableNames(BaseModel):
        not_score: float = 0.5
        not_value: float = 0.3

    other_variable_names = create_aggregation_overview(OtherVariableNames())

    class SameNameOtherClassAggregation(BaseModel):
        not_score: float = 0.5
        not_value: float = 0.3

    same_variable_names_other_class = create_aggregation_overview(
        SameNameOtherClassAggregation()
    )

    # when then
    with pytest.raises(ValueError):
        df = aggregation_overviews_to_pandas([overview, other_variable_names])
    df = aggregation_overviews_to_pandas([overview, other_variable_names], strict=False)

    with pytest.raises(ValueError):
        df = aggregation_overviews_to_pandas(
            [overview, same_variable_names_other_class]
        )
    df = aggregation_overviews_to_pandas(
        [overview, same_variable_names_other_class], strict=False
    )

    assert len(df) == 2
