from pydantic import BaseModel

from intelligence_layer.core import Task, Tracer
from intelligence_layer.evaluation import (
    Example,
    IncrementalEvaluationLogic,
    IncrementalEvaluator,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    Runner,
    SuccessfulExampleOutput,
)


class DummyEvaluation(BaseModel):
    all_run_ids: list[str]
    old_run_ids: list[list[str]]


class DummyIncrementalLogic(IncrementalEvaluationLogic[str, str, str, DummyEvaluation]):
    def __init__(self) -> None:
        super().__init__()

    def do_incremental_evaluate(
        self,
        example: Example[str, str],
        outputs: list[SuccessfulExampleOutput[str]],
        already_evaluated_outputs: list[list[SuccessfulExampleOutput[str]]],
    ) -> DummyEvaluation:
        return DummyEvaluation(
            all_run_ids=[output.run_id for output in outputs],
            old_run_ids=[
                [output.run_id for output in evaluated_output]
                for evaluated_output in already_evaluated_outputs
            ],
        )


class DummyTask(Task[str, str]):
    def __init__(self, info: str) -> None:
        super().__init__()
        self._info = info

    def do_run(self, input: str, tracer: Tracer) -> str:
        return f"{input} {self._info}"


def test_incremental_evaluator_separates_all_runs_and_previous_runs() -> None:
    # Given
    examples = [Example(input="a", expected_output="0", id="id_0")]
    dataset_repository = InMemoryDatasetRepository()
    run_repository = InMemoryRunRepository()
    evaluation_repository = InMemoryEvaluationRepository()
    dataset = dataset_repository.create_dataset(
        examples=examples, dataset_name="test_examples"
    )

    evaluator = IncrementalEvaluator(
        dataset_repository=dataset_repository,
        run_repository=run_repository,
        evaluation_repository=evaluation_repository,
        description="test_incremental_evaluator",
        incremental_evaluation_logic=DummyIncrementalLogic(),
    )

    def create_run(name: str) -> str:
        runner = Runner(
            task=DummyTask(name),
            dataset_repository=dataset_repository,
            run_repository=run_repository,
            description=f"Runner of {name}",
        )
        return runner.run_dataset(dataset.id).id

    first_run_id = create_run("first")

    first_evaluation_overview = evaluator.evaluate_additional_runs(first_run_id)

    second_run_id = create_run("second")

    second_evaluation_overview = evaluator.evaluate_additional_runs(
        first_run_id,
        second_run_id,
        previous_evaluation_ids=[first_evaluation_overview.id],
    )

    second_result = next(
        iter(evaluator.evaluation_lineages(second_evaluation_overview.id))
    ).evaluation.result
    assert isinstance(second_result, DummyEvaluation)
    assert sorted(second_result.all_run_ids) == sorted([first_run_id, second_run_id])
    assert sorted(second_result.old_run_ids) == sorted([[first_run_id]])

    independent_run_id = create_run("independent")

    independent_evaluation_overview = evaluator.evaluate_additional_runs(
        independent_run_id
    )

    third_run_id = create_run("third")

    third_evaluation_overview = evaluator.evaluate_additional_runs(
        first_run_id,
        second_run_id,
        independent_run_id,
        third_run_id,
        previous_evaluation_ids=[
            second_evaluation_overview.id,
            independent_evaluation_overview.id,
        ],
    )

    third_result = next(
        iter(evaluator.evaluation_lineages(third_evaluation_overview.id))
    ).evaluation.result
    assert isinstance(third_result, DummyEvaluation)
    assert sorted(third_result.all_run_ids) == sorted(
        [first_run_id, second_run_id, independent_run_id, third_run_id]
    )
    assert sorted(third_result.old_run_ids[0]) == sorted([first_run_id, second_run_id])
    assert sorted(third_result.old_run_ids[1]) == sorted([independent_run_id])
