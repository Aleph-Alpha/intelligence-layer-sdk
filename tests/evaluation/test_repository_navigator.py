
import pytest
from pydantic import BaseModel

from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer.tracer import TaskSpan
from intelligence_layer.evaluation.dataset.domain import Example
from intelligence_layer.evaluation.dataset.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
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
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput
from intelligence_layer.evaluation.run.in_memory_run_repository import (
    InMemoryRunRepository,
)
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


def test_works_on_run_overviews() -> None:
    dataset_repository = InMemoryDatasetRepository()
    dataset = dataset_repository.create_dataset(
        [
            DummyExample(
                input="input0", expected_output="expected_output0", data="data0"
            ),
            DummyExample(
                input="input1", expected_output="expected_output1", data="data1"
            ),
        ],
        "test",
    )

    run_repository = InMemoryRunRepository()
    task = DummyTask()
    run_overview = Runner(
        task, dataset_repository, run_repository, "Runner"
    ).run_dataset(dataset.id)

    x = RepositoryNavigator(dataset_repository, run_repository)

    # when
    res = list(x.run_data(run_overview.id, str, str, str))

    # then
    res = sorted(res, key=lambda result: result.example.input)
    for i in range(2):
        assert res[i].example.input == f"input{i}"
        assert res[i].example.expected_output == f"expected_output{i}"
        assert res[i].output.output == f"input{i} -> output"


def test_works_on_evaluation() -> None:
    dataset_repository = InMemoryDatasetRepository()
    dataset = dataset_repository.create_dataset(
        [
            DummyExample(
                input="input0", expected_output="expected_output0", data="data0"
            ),
            DummyExample(
                input="input1", expected_output="expected_output1", data="data1"
            ),
        ],
        "test",
    )

    run_repository = InMemoryRunRepository()
    task = DummyTask()
    run_overview = Runner(
        task, dataset_repository, run_repository, "Runner"
    ).run_dataset(dataset.id)

    eval_repository = InMemoryEvaluationRepository()
    eval_logic = DummyEvalLogic()
    eval_overview = Evaluator(
        dataset_repository, run_repository, eval_repository, "Evaluator", eval_logic
    ).evaluate_runs(run_overview.id)

    x = RepositoryNavigator(dataset_repository, run_repository, eval_repository)

    # when
    res = list(x.eval_data(eval_overview.id, str, str, str, DummyEval))

    # then
    res = sorted(res, key=lambda result: result.example.input)
    for i in range(2):
        assert res[i].example.input == f"input{i}"
        assert res[i].example.expected_output == f"expected_output{i}"
        assert res[i].output.output == f"input{i} -> output"
        eval_result = res[i].evaluation.result
        assert isinstance(eval_result, DummyEval)
        assert eval_result.eval.startswith(f"input{i}")


def test_initialization_gives_warning_if_not_compatible() -> None:
    dataset_repository = InMemoryDatasetRepository()
    run_repository = InMemoryRunRepository()

    x = RepositoryNavigator(dataset_repository, run_repository)
    with pytest.raises(ValueError):
        list(x.eval_data("non-existent", str, str, str, DummyEval))
