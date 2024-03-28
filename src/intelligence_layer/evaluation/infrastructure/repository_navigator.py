import itertools
from typing import Generic, Iterable, Sequence

from pydantic import BaseModel

from intelligence_layer.core.task import Input, Output
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    ExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.run.domain import ExampleOutput
from intelligence_layer.evaluation.run.run_repository import RunRepository


class RunLineage(BaseModel, Generic[Input, ExpectedOutput, Output]):
    example: Example[Input, ExpectedOutput]
    output: ExampleOutput[Output]


class EvalLineage(BaseModel, Generic[Input, ExpectedOutput, Output, Evaluation]):
    example: Example[Input, ExpectedOutput]
    outputs: Sequence[ExampleOutput[Output]]
    evaluation: ExampleEvaluation[Evaluation]


class RepositoryNavigator:
    """The `RepositoryNavigator` is used to retrieve coupled data from multiple repositories."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: EvaluationRepository | None = None,
    ) -> None:
        self._dataset_repository = dataset_repository
        self._run_repository = run_repository
        self._eval_repository = evaluation_repository

    def run_data(
        self,
        run_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
    ) -> Iterable[RunLineage[Input, ExpectedOutput, Output]]:
        run_overview = self._run_repository.run_overview(run_id)
        if run_overview is None:
            return []

        examples = list(
            self._dataset_repository.examples(
                run_overview.dataset_id,
                input_type,
                expected_output_type,
            )
        )

        example_outputs = list(
            self._run_repository.example_outputs(run_id, output_type)
        )

        # join
        for example, example_output in itertools.product(examples, example_outputs):
            if example.id == example_output.example_id:
                yield RunLineage(example=example, output=example_output)

    def eval_data(
        self,
        eval_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
        evaluation_type: type[Evaluation],
    ) -> Iterable[EvalLineage[Input, ExpectedOutput, Output, Evaluation]]:
        if self._eval_repository is None:
            raise ValueError("Evaluation Repository is not set, but required.")
        eval_overview = self._eval_repository.evaluation_overview(eval_id)
        if eval_overview is None:
            return []

        evaluations = list(
            self._eval_repository.example_evaluations(eval_id, evaluation_type)
        )
        run_lineages = itertools.chain.from_iterable(
            self.run_data(overview.id, input_type, expected_output_type, output_type)
            for overview in eval_overview.run_overviews
        )

        # join
        for run_lineage, evaluation in itertools.product(run_lineages, evaluations):
            if run_lineage.example.id == evaluation.example_id:
                yield EvalLineage(
                    example=run_lineage.example,
                    output=run_lineage.output,
                    evaluation=evaluation,
                )

    def run_single_example(
        self,
        run_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
    ) -> RunLineage[Input, ExpectedOutput, Output] | None:

        run_overview = self._run_repository.run_overview(run_id)
        if run_overview is None:
            return None

        example = self._dataset_repository.example(
            run_overview.dataset_id, example_id, input_type, expected_output_type
        )
        example_output = self._run_repository.example_output(
            run_id, example_id, output_type
        )

        return RunLineage(example=example, output=example_output)

    def eval_single_example(
        self,
        eval_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
        evaluation_type: type[Evaluation],
    ) -> Sequence[EvalLineage[Input, ExpectedOutput, Output, Evaluation]] | None:

        eval_overview = self._eval_repository.evaluation_overview(eval_id)
        if eval_overview is None:
            return None

        run_lineages = [
            self.run_single_example(
                overview.id, example_id, input_type, expected_output_type, output_type
            )
            for overview in eval_overview.run_overviews
        ]

        example_evaluation = self._eval_repository.example_evaluation(
            eval_id, example_id, evaluation_type
        )

        return EvalLineage(
            example=example, output=example_output, evaluation=example_evaluation
        )
