import itertools
from collections.abc import Iterable, Sequence
from typing import Generic, Optional

import pandas as pd
import rich
from rich.tree import Tree

from intelligence_layer.core import Input, Output, Tracer
from intelligence_layer.evaluation.aggregation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    ExampleEvaluation,
    FailedExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.run.domain import ExampleOutput, FailedExampleRun
from intelligence_layer.evaluation.run.run_repository import RunRepository

EXAMPLE_OUTPUT_TYPE = ExampleOutput[Output] | ExampleOutput[FailedExampleRun]


class RunLineage(Generic[Input, ExpectedOutput, Output]):
    example: Example[Input, ExpectedOutput]
    output: EXAMPLE_OUTPUT_TYPE
    tracer: Optional[Tracer]

    def __init__(
        self,
        example: Example[Input, ExpectedOutput],
        output: EXAMPLE_OUTPUT_TYPE,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.example = example
        self.output = output
        self.tracer = tracer

    def _rich_render(self) -> Tree:
        tree = Tree("Run Lineage")
        tree.add(self.example._rich_render())
        tree.add(self.output._rich_render(skip_example_id=True))
        return tree

    def _ipython_display_(self) -> None:
        rich.print(self._rich_render())


def run_lineages_to_pandas(
    run_lineages: Sequence[RunLineage[Input, ExpectedOutput, Output]],
) -> pd.DataFrame:
    """Converts a sequence of `RunLineage` objects to a pandas `DataFrame`.

    The `RunLineage` objects are stored in the column `"lineage"`.
    The `DataFrame` is indexed by `(example_id, run_id)`.

    Args:
        run_lineages: The lineages to convert.

    Returns:
        A pandas `DataFrame` with the data contained in the `run_lineages`.
    """
    df = pd.DataFrame(
        [
            vars(lineage.example) | vars(lineage.output) | {"lineage": lineage}
            for lineage in run_lineages
        ]
    )
    df = df.drop(columns="id")
    df = df.set_index(["example_id", "run_id"])
    return df


EXAMPLE_EVAL_TYPE = (
    ExampleEvaluation[Evaluation] | ExampleEvaluation[FailedExampleEvaluation]
)


class EvaluationLineage(Generic[Input, ExpectedOutput, Output, Evaluation]):
    example: Example[Input, ExpectedOutput]
    outputs: Sequence[EXAMPLE_OUTPUT_TYPE]
    evaluation: EXAMPLE_EVAL_TYPE
    tracers: Sequence[Optional[Tracer]]

    def __init__(
        self,
        example: Example[Input, ExpectedOutput],
        outputs: Sequence[EXAMPLE_OUTPUT_TYPE],
        evaluation: EXAMPLE_EVAL_TYPE,
        tracers: Sequence[Optional[Tracer]],
    ) -> None:
        self.example = example
        self.outputs = outputs
        self.evaluation = evaluation
        self.tracers = tracers

    def _rich_render(self) -> Tree:
        tree = Tree("Run Lineage")
        tree.add(self.example._rich_render())
        output_tree = Tree("Outputs")
        for output in self.outputs:
            output_tree.add(output._rich_render(skip_example_id=True))
        tree.add(output_tree)
        tree.add(self.evaluation._rich_render(skip_example_id=True))
        return tree

    def _ipython_display_(self) -> None:
        rich.print(self._rich_render())


def evaluation_lineages_to_pandas(
    evaluation_lineages: Sequence[
        EvaluationLineage[Input, ExpectedOutput, Output, Evaluation]
    ],
) -> pd.DataFrame:
    """Converts a sequence of `EvaluationLineage` objects to a pandas `DataFrame`.

    The `EvaluationLineage` objects are stored in the column `"lineage"`.
    The `DataFrame` is indexed by `(example_id, evaluation_id, run_id)`.
    Each `output` of every lineage will contribute one row in the `DataFrame`.

    Args:
        evaluation_lineages: The lineages to convert.

    Returns:
        A pandas `DataFrame` with the data contained in the `evaluation_lineages`.
    """
    df = pd.DataFrame(
        [
            vars(lineage.example)
            | vars(output)
            | vars(lineage.evaluation)
            | {"tracer": lineage.tracers[index]}
            | {"lineage": lineage}
            for lineage in evaluation_lineages
            for index, output in enumerate(lineage.outputs)
        ]
    )
    df = df.drop(columns="id")
    df = df.set_index(["example_id", "evaluation_id", "run_id"])
    return df


def aggregation_overviews_to_pandas(
    aggregation_overviews: Sequence[AggregationOverview[AggregatedEvaluation]],
    unwrap_statistics: bool = True,
    strict: bool = True,
    unwrap_metadata: bool = True,
) -> pd.DataFrame:
    """Converts aggregation overviews to a pandas table for easier comparison.

    Args:
        aggregation_overviews: Overviews to convert.
        unwrap_statistics: Unwrap the `statistics` field in the overviews into separate columns.
            Defaults to True.
        strict: Allow only overviews with exactly equal `statistics` types. Defaults to True.
        unwrap_metadata: Unwrap the `metadata` field in the overviews into separate columns.
            Defaults to True.

    Returns:
        A pandas :class:`DataFrame` containing an overview per row with fields as columns.
    """
    overviews = list(aggregation_overviews)
    if strict and len(overviews) > 1:
        first_type = overviews[0].statistics.__class__
        if any(
            overview.statistics.__class__ != first_type for overview in overviews[1:]
        ):
            raise ValueError(
                "Aggregation overviews contain different types, which is not allowed with strict=True"
            )

    df = pd.DataFrame(
        [model.model_dump(mode="json") for model in aggregation_overviews]
    )
    if unwrap_statistics and "statistics" in df.columns:
        df = df.join(pd.DataFrame(df["statistics"].to_list())).drop(
            columns=["statistics"]
        )
    if unwrap_metadata and "metadata" in df.columns:
        df = pd.concat([df, pd.json_normalize(df["metadata"])], axis=1).drop(  # type: ignore
            columns=["metadata"]
        )

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

        Yields:
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
                yield RunLineage(
                    example=example,
                    output=example_output,
                    tracer=self._run_repository.example_tracer(
                        run_id=run_id, example_id=example.id
                    ),
                )

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

        Yields:
            All :class:`EvaluationLineage`s for the given evaluation id.
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
            tracers = []
            for run_lineage in run_lineages:
                if run_lineage.example.id == evaluation.example_id:
                    if example is None:
                        # the evaluation has only one example
                        # and all relevant run lineages contain the same example
                        example = run_lineage.example
                    outputs.append(run_lineage.output)
                    tracers.append(
                        self._run_repository.example_tracer(
                            run_lineage.output.run_id, run_lineage.output.example_id
                        )
                    )

            if example is not None:
                yield EvaluationLineage(
                    example=example,
                    outputs=outputs,
                    evaluation=evaluation,
                    tracers=tracers,
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

        return RunLineage(
            example=example,
            output=example_output,
            tracer=self._run_repository.example_tracer(run_id, example_id),
        )

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
            example_id: The id of the example of interest
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
            tracers=[lineage.tracer for lineage in existing_run_lineages],
        )
