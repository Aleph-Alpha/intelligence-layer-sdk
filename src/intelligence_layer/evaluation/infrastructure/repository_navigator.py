import itertools
from typing import Generic, Iterable, Sequence

import pandas as pd
import rich
from pydantic import BaseModel
from rich.tree import Tree

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

    def _ipython_display_(self):
        tree = Tree("Run Lineage")
        tree.add(self.example._rich_render())
        tree.add(self.output._rich_render(skip_example_id=True))
        rich.print(tree)


def run_lineages_to_pandas(
    evaluation_lineages: Sequence[RunLineage[Input, ExpectedOutput, Output]]
):
    df = pd.DataFrame(
        [
            vars(lineage.example) | vars(lineage.output) | {"lineage": lineage}
            for lineage in evaluation_lineages
        ]
    )
    df = df.drop(columns="id")
    df = df.set_index(["example_id", "run_id"])
    return df


class EvaluationLineage(BaseModel, Generic[Input, ExpectedOutput, Output, Evaluation]):
    example: Example[Input, ExpectedOutput]
    outputs: Sequence[ExampleOutput[Output]]
    evaluation: ExampleEvaluation[Evaluation]

    def _ipython_display_(self):
        tree = Tree("Run Lineage")
        tree.add(self.example._rich_render())
        output_tree = Tree("Outputs")
        for output in self.outputs:
            output_tree.add(output._rich_render(skip_example_id=True))
        tree.add(output_tree)
        tree.add(self.evaluation._rich_render(skip_example_id=True))
        rich.print(tree)


def evaluation_lineages_to_pandas(
    evaluation_lineages: Sequence[
        EvaluationLineage[Input, ExpectedOutput, Output, Evaluation]
    ]
):
    df = pd.DataFrame(
        [
            vars(lineage.example)
            | vars(output)
            | vars(lineage.evaluation)
            | {"lineage": lineage}
            for lineage in evaluation_lineages
            for output in lineage.outputs
        ]
    )
    df = df.drop(columns="id")
    df = df.set_index(["example_id", "evaluation_id", "run_id"])
    return df


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

    def run_lineages(
        self,
        run_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
    ) -> Iterable[RunLineage[Input, ExpectedOutput, Output]]:
        """Retrieves all :class:`RunLineage`s for the run with id `run_id`.

        Args:
            run_id: The id of the run
            input_type: The type of the input as defined by the :class:`Example`
            expected_output_type: The type of the expected output as defined by the :class:`Example`
            output_type: The type of the run output as defined by the :class:`Output`

        Returns:
            An iterator over all :class:`RunLineage`s for the given run id.
        """
        run_overview = self._run_repository.run_overview(run_id)
        if run_overview is None:
            raise ValueError(f"Run repository does not contain a run with id {run_id}.")

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

    def evaluation_lineages(
        self,
        evaluation_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
        evaluation_type: type[Evaluation],
    ) -> Iterable[EvaluationLineage[Input, ExpectedOutput, Output, Evaluation]]:
        """Retrieves all :class:`EvaluationLineage`s for the evaluation with id `evaluation_id`.

        Args:
            evaluation_id: The id of the evaluation
            input_type: The type of the input as defined by the :class:`Example`
            expected_output_type: The type of the expected output as defined by the :class:`Example`
            output_type: The type of the run output as defined by the :class:`Output`
            evaluation_type: The type of the evaluation as defined by the :class:`Evaluation`

        Returns:
            An iterator over all :class:`EvaluationLineage`s for the given evaluation id.
        """
        if self._eval_repository is None:
            raise ValueError("Evaluation Repository is not set, but required.")

        eval_overview = self._eval_repository.evaluation_overview(evaluation_id)
        if eval_overview is None:
            raise ValueError(
                f"Evaluation repository does not contain an evaluation with id {evaluation_id}."
            )

        evaluations = list(
            self._eval_repository.example_evaluations(evaluation_id, evaluation_type)
        )
        run_lineages = list(
            itertools.chain.from_iterable(
                self.run_lineages(
                    overview.id, input_type, expected_output_type, output_type
                )
                for overview in eval_overview.run_overviews
            )
        )

        # join
        for evaluation in evaluations:
            example = None
            outputs = []
            for run_lineage in run_lineages:
                if run_lineage.example.id == evaluation.example_id:
                    if example is None:
                        # the evaluation has only one example
                        # and all relevant run lineages contain the same example
                        example = run_lineage.example
                    outputs.append(run_lineage.output)

            if example is not None:
                yield EvaluationLineage(
                    example=example, outputs=outputs, evaluation=evaluation
                )

    def run_lineage(
        self,
        run_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
    ) -> RunLineage[Input, ExpectedOutput, Output] | None:
        """Retrieves the :class:`RunLineage` for the run with id `run_id` and example with id `example_id`.

        Args:
            run_id: The id of the run
            example_id: The id of the example
            input_type: The type of the input as defined by the :class:`Example`
            expected_output_type: The type of the expected output as defined by the :class:`Example`
            output_type: The type of the run output as defined by the :class:`Output`

        Returns:
            The :class:`RunLineage` for the given run id and example id, `None` if the example or an output for the example does not exist.
        """

        run_overview = self._run_repository.run_overview(run_id)
        if run_overview is None:
            raise ValueError(f"Run repository does not contain a run with id {run_id}.")

        example = self._dataset_repository.example(
            run_overview.dataset_id, example_id, input_type, expected_output_type
        )
        if example is None:
            return None

        example_output = self._run_repository.example_output(
            run_id, example_id, output_type
        )
        if example_output is None:
            return None

        return RunLineage(example=example, output=example_output)

    def evaluation_lineage(
        self,
        evaluation_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        output_type: type[Output],
        evaluation_type: type[Evaluation],
    ) -> EvaluationLineage[Input, ExpectedOutput, Output, Evaluation] | None:
        """Retrieves the :class:`EvaluationLineage` for the evaluation with id `evaluation_id` and example with id `example_id`.

        Args:
            evaluation_id: The id of the evaluation
            example_id: The id of the example
            input_type: The type of the input as defined by the :class:`Example`
            expected_output_type: The type of the expected output as defined by the :class:`Example`
            output_type: The type of the run output as defined by the :class:`Output`
            evaluation_type: The type of the evaluation as defined by the :class:`Evaluation`

        Returns:
            The :class:`EvaluationLineage` for the given evaluation id and example id.
            Returns `None` if the lineage is not complete because either an example, a run, or an evaluation does not exist.
        """

        if self._eval_repository is None:
            raise ValueError("Evaluation Repository is not set, but required.")

        eval_overview = self._eval_repository.evaluation_overview(evaluation_id)
        if eval_overview is None:
            raise ValueError(
                f"Evaluation repository does not contain an evaluation with id {evaluation_id}."
            )

        run_lineages = [
            self.run_lineage(
                overview.id, example_id, input_type, expected_output_type, output_type
            )
            for overview in eval_overview.run_overviews
        ]
        existing_run_lineages = [
            lineage for lineage in run_lineages if lineage is not None
        ]
        if len(existing_run_lineages) == 0:
            return None

        example_evaluation = self._eval_repository.example_evaluation(
            evaluation_id, example_id, evaluation_type
        )
        if example_evaluation is None:
            return None

        return EvaluationLineage(
            example=existing_run_lineages[0].example,
            outputs=[lineage.output for lineage in existing_run_lineages],
            evaluation=example_evaluation,
        )
