from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.evaluation import (
    AggregationLogic,
    ComparisonEvaluation,
    Dataset,
    DatasetRepository,
    EloEvaluationLogic,
    EvaluationLogic,
    EvaluationOverview,
    Evaluator,
    Example,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    Matches,
    MatchOutcome,
    Runner,
    RunOverview,
    RunRepository,
    SingleOutputEvaluationLogic,
    SuccessfulExampleOutput,
)


class DummyExample(Example[str, str]):
    data: str


class DummyEvaluation(BaseModel):
    eval: str


class DummyAggregation(BaseModel):
    num_evaluations: int


class DummyTask(Task[str, str]):
    def do_run(self, input: str, task_span: TaskSpan) -> str:
        return f"{input} -> output"


class DummyEvaluationLogic(EvaluationLogic[str, str, str, DummyEvaluation]):
    def do_evaluate(
        self, example: Example[str, str], *output: SuccessfulExampleOutput[str]
    ) -> DummyEvaluation:
        output_str = "(" + (", ".join(o.output for o in output)) + ")"
        return DummyEvaluation(
            eval=f"{example.input}, {example.expected_output}, {output_str} -> evaluation"
        )


class DummyAggregationLogic(AggregationLogic[DummyEvaluation, DummyAggregation]):
    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> DummyAggregation:
        return DummyAggregation(num_evaluations=len(list(evaluations)))


class ComplexDummyInput(BaseModel):
    input_data: dict[str, int]


class ComplexDummyExpectedOutput(BaseModel):
    expected_output: dict[str, int]


class ComplexDummyExample(Example[ComplexDummyInput, ComplexDummyExpectedOutput]):
    pass


class ComplexDummyOutput(BaseModel):
    output: str
    input: ComplexDummyInput


class ComplexDummyEvaluation(BaseModel):
    input: ComplexDummyInput
    expected_output: ComplexDummyExpectedOutput
    output: ComplexDummyOutput


class ComplexDummyAggregation(BaseModel):
    evals: list[ComplexDummyEvaluation]


class ComplexDummyTask(Task[ComplexDummyInput, ComplexDummyOutput]):
    def do_run(
        self, input: ComplexDummyInput, task_span: TaskSpan
    ) -> ComplexDummyOutput:
        return ComplexDummyOutput(output="output", input=input)


class ComplexDummyEvaluationLogic(
    SingleOutputEvaluationLogic[
        ComplexDummyInput,
        ComplexDummyOutput,
        ComplexDummyExpectedOutput,
        ComplexDummyEvaluation,
    ]
):
    def do_evaluate_single_output(
        self,
        example: Example[ComplexDummyInput, ComplexDummyExpectedOutput],
        output: ComplexDummyOutput,
    ) -> ComplexDummyEvaluation:
        return ComplexDummyEvaluation(
            input=example.input, expected_output=example.expected_output, output=output
        )


class ComplexDummyAggregationLogic(
    AggregationLogic[ComplexDummyEvaluation, ComplexDummyAggregation]
):
    def aggregate(
        self, evaluations: Iterable[ComplexDummyEvaluation]
    ) -> ComplexDummyAggregation:
        return ComplexDummyAggregation(evals=list(evaluations))


class DummyTaskCanFail(Task[str, str]):
    EXAMPLE_FAILURE_INPUT = "input1"

    def __init__(self) -> None:
        super().__init__()
        self._raise_exception = True

    def do_run(self, input: str, task_span: TaskSpan) -> str:
        if input == self.EXAMPLE_FAILURE_INPUT and self._raise_exception:
            self._raise_exception = False
            raise Exception("Some random failure in the system.")
        return f"{input} -> output"


class DummyEloEvaluationLogic(EloEvaluationLogic[str, str, str]):
    def grade(
        self,
        first: SuccessfulExampleOutput[str],
        second: SuccessfulExampleOutput[str],
        example: Example[str, str],
    ) -> MatchOutcome:
        return MatchOutcome.DRAW

    def do_incremental_evaluate(
        self,
        example: Example[str, str],
        outputs: list[SuccessfulExampleOutput[str]],
        already_evaluated_outputs: list[list[SuccessfulExampleOutput[str]]],
    ) -> Matches:
        player_a = SuccessfulExampleOutput(run_id="1", example_id="1", output="1")
        player_b = SuccessfulExampleOutput(run_id="2", example_id="2", output="2")
        return Matches(
            comparison_evaluations=[
                ComparisonEvaluation(
                    first_player="1",
                    second_player="2",
                    outcome=self.grade(player_a, player_b, example),
                )
            ]
        )


class ExampleData:
    examples: Sequence[DummyExample]
    complex_examples: Sequence[ComplexDummyExample]
    dataset_repository: InMemoryDatasetRepository
    run_repository: InMemoryRunRepository
    evaluation_repository: InMemoryEvaluationRepository
    runner: Runner[str, str]
    evaluator: Evaluator[str, str, str, DummyEvaluation]
    dataset: Dataset
    run_overview_1: RunOverview
    run_overview_2: RunOverview
    evaluation_overview_1: EvaluationOverview
    evaluation_overview_2: EvaluationOverview
    studio_project_name: str


def example_data() -> ExampleData:
    examples = [
        DummyExample(input="input0", expected_output="expected_output0", data="data0"),
        DummyExample(
            input=DummyTaskCanFail.EXAMPLE_FAILURE_INPUT,
            expected_output="expected_output1",
            data="data1",
        ),
    ]
    complex_examples = [
        ComplexDummyExample(
            input=ComplexDummyInput(input_data={"a": 1, "b": 2}),
            expected_output=ComplexDummyExpectedOutput(
                expected_output={"c": 3, "d": 4}
            ),
        ),
        ComplexDummyExample(
            input=ComplexDummyInput(input_data={"x": 10, "y": 20}),
            expected_output=ComplexDummyExpectedOutput(
                expected_output={"p": 30, "q": 40}
            ),
        ),
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
    evaluation_overview_1 = evaluator.evaluate_runs(
        run_overview_1.id, run_overview_2.id
    )
    evaluation_overview_2 = evaluator.evaluate_runs(
        run_overview_1.id, run_overview_2.id
    )

    example_data = ExampleData()
    example_data.examples = examples
    example_data.complex_examples = complex_examples
    example_data.dataset_repository = dataset_repository
    example_data.run_repository = run_repository
    example_data.evaluation_repository = evaluation_repository
    example_data.runner = runner
    example_data.evaluator = evaluator
    example_data.dataset = dataset
    example_data.run_overview_1 = run_overview_1
    example_data.run_overview_2 = run_overview_2
    example_data.evaluation_overview_1 = evaluation_overview_1
    example_data.evaluation_overview_2 = evaluation_overview_2
    example_data.studio_project_name = "Example Project"

    return example_data


class StoryTaskInput(BaseModel):  # Should already be implemented in your task
    topic: str
    targeted_word_count: int


class StoryTaskOutput(BaseModel):  # Should already be implemented in your task
    story: str


class StoryTask(Task[StoryTaskInput, StoryTaskOutput]):
    def do_run(self, input: StoryTaskInput, task_span: TaskSpan) -> StoryTaskOutput:
        return StoryTaskOutput(
            story=f"cool story about {input.topic} with {input.targeted_word_count} words"
        )


class ArgillaExampleData:
    dataset_repository: DatasetRepository
    run_repository: RunRepository
    run_ids: list[str]


def argilla_example_data() -> ArgillaExampleData:
    dataset_repository = InMemoryDatasetRepository()
    run_repository = InMemoryRunRepository()
    examples = [
        Example(
            input=StoryTaskInput(topic="sports", targeted_word_count=10),
            expected_output=None,
        ),
        Example(
            input=StoryTaskInput(topic="sports", targeted_word_count=100),
            expected_output=None,
        ),
    ]
    dataset = dataset_repository.create_dataset(
        examples=examples, dataset_name="StoryTasks"
    )
    run_overview_1 = Runner(
        StoryTask(), dataset_repository, run_repository, "StoryTaskEval"
    ).run_dataset(dataset.id)
    run_overview_2 = Runner(
        StoryTask(), dataset_repository, run_repository, "StoryTaskEval2"
    ).run_dataset(dataset.id)

    example_data = ArgillaExampleData()
    example_data.dataset_repository = dataset_repository
    example_data.run_repository = run_repository
    example_data.run_ids = [run_overview_1.id, run_overview_2.id]
    return example_data
