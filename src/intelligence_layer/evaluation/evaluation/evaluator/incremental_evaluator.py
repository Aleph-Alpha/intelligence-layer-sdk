from abc import abstractmethod
from collections.abc import Sequence
from enum import Enum
from itertools import combinations
from typing import Optional

from pydantic import BaseModel

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import (
    EvaluationLogic,
    Evaluator,
)
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput
from intelligence_layer.evaluation.run.run_repository import RunRepository


class IncrementalEvaluationLogic(
    EvaluationLogic[Input, Output, ExpectedOutput, Evaluation]
):
    def __init__(self) -> None:
        super().__init__()
        self._previous_run_output_ids: list[set[str]] = []

    def set_previous_run_output_ids(
        self, previous_run_output_ids: list[set[str]]
    ) -> None:
        self._previous_run_output_ids = previous_run_output_ids

    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> Evaluation:
        """Executes the evaluation for this specific example.

        Responsible for comparing the input & expected output of a task to the
        actually generated output. The difference to the standard :class:`EvaluationLogic`'s `do_evaluate` is that
        this method will separate already processed evaluation from new ones before handing them over to
        `do_incremental_evaluate`.

        Args:
            example: Input data of :class:`Task` to produce the output.
            *output: Outputs of the :class:`Task`.

        Returns:
            :class:`Evaluation`: The metrics that come from the evaluated :class:`Task`.
        """
        already_evaluated_outputs = []
        for run_output_ids in self._previous_run_output_ids:
            already_evaluated_outputs.append(
                [
                    current_output
                    for current_output in output
                    if current_output.run_id in run_output_ids
                ]
            )

        return self.do_incremental_evaluate(
            example, list(output), already_evaluated_outputs
        )

    @abstractmethod
    def do_incremental_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        outputs: list[SuccessfulExampleOutput[Output]],
        already_evaluated_outputs: list[list[SuccessfulExampleOutput[Output]]],
    ) -> Evaluation:
        pass


class IncrementalEvaluator(Evaluator[Input, Output, ExpectedOutput, Evaluation]):
    """:class:`Evaluator` for evaluating additional runs on top of previous evaluations. Intended for use with :class:`IncrementalEvaluationLogic`.

    Args:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        description: Human-readable description for the evaluator.
        incremental_evaluation_logic: The logic to use for evaluation.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: EvaluationRepository,
        description: str,
        incremental_evaluation_logic: IncrementalEvaluationLogic[
            Input, Output, ExpectedOutput, Evaluation
        ],
    ) -> None:
        super().__init__(
            dataset_repository=dataset_repository,
            run_repository=run_repository,
            evaluation_repository=evaluation_repository,
            description=description,
            evaluation_logic=incremental_evaluation_logic,
        )
        self._evaluation_logic: IncrementalEvaluationLogic[
            Input, Output, ExpectedOutput, Evaluation
        ]

    def evaluate_additional_runs(
        self,
        *run_ids: str,
        previous_evaluation_ids: Optional[list[str]] = None,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
        labels: Optional[set[str]] = None,
        metadata: Optional[SerializableDict] = None,
    ) -> EvaluationOverview:
        """Evaluate all runs while considering which runs have already been evaluated according to `previous_evaluation_id`.

        For each set of successful outputs in the referenced runs,
        :func:`EvaluationLogic.do_evaluate` is called and eval metrics are produced &
        stored in the provided :class:`EvaluationRepository`.

        Args:
            *run_ids: The runs to be evaluated. Each run is expected to have the same
                dataset as input (which implies their tasks have the same input-type)
                and their tasks have the same output-type. For each example in the
                dataset referenced by the runs the outputs of all runs are collected
                and if all of them were successful they are passed on to the implementation
                specific evaluation. The method compares all run of the provided ids to each other.
            previous_evaluation_ids: IDs of previous evaluation to consider
            num_examples: The number of examples which should be evaluated from the given runs.
                Always the first n runs stored in the evaluation repository. Defaults to None.
            abort_on_error: Flag to abort all evaluations when an error occurs. Defaults to False.
            labels: A list of labels for filtering. Defaults to an empty list.
            metadata: A dict for additional information about the evaluation overview. Defaults to an empty dict.

        Returns:
            EvaluationOverview: An overview of the evaluation. Individual :class:`Evaluation`s will not be
            returned but instead stored in the :class:`EvaluationRepository` provided in the
            __init__.
        """
        if metadata is None:
            metadata = dict()
        if labels is None:
            labels = set()
        previous_run_ids = []
        previous_evaluation_ids = previous_evaluation_ids or []

        for previous_evaluation_id in previous_evaluation_ids:
            prev_run_ids: set[str] = set()
            lineages = self.evaluation_lineages(previous_evaluation_id)
            for lineage in lineages:
                for output in lineage.outputs:
                    prev_run_ids.add(output.run_id)
            previous_run_ids.append(prev_run_ids)

        self._evaluation_logic.set_previous_run_output_ids(previous_run_ids)
        return super().evaluate_runs(
            *run_ids,
            num_examples=num_examples,
            abort_on_error=abort_on_error,
            labels=labels,
            metadata=metadata,
        )

    def evaluate_runs(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
        skip_example_on_any_failure: bool = True,
        description: Optional[str] = None,
        labels: set[str] | None = None,
        metadata: SerializableDict | None = None,
    ) -> EvaluationOverview:
        if metadata is None:
            metadata = dict()
        if labels is None:
            labels = set()
        self._evaluation_logic.set_previous_run_output_ids([])
        return super().evaluate_runs(
            *run_ids,
            num_examples=num_examples,
            skip_example_on_any_failure=skip_example_on_any_failure,
            abort_on_error=abort_on_error,
            description=description,
        )


class MatchOutcome(str, Enum):
    A_WINS = "a_wins"
    DRAW = "draw"
    B_WINS = "b_wins"

    @property
    def payoff(self) -> tuple[float, float]:
        if self == self.A_WINS:
            return (1, 0)
        if self == self.DRAW:
            return (0.5, 0.5)
        return (0, 1)

    @staticmethod
    def from_rank_literal(rank: int) -> "MatchOutcome":
        match rank:
            case 1:
                return MatchOutcome.A_WINS
            case 2:
                return MatchOutcome.B_WINS
            case 3:
                return MatchOutcome.DRAW
            case _:
                raise ValueError(f"Got unexpected rank {rank}")


class ComparisonEvaluation(BaseModel):
    first_player: str
    second_player: str
    outcome: MatchOutcome


class Matches(BaseModel):
    comparison_evaluations: Sequence[ComparisonEvaluation]


class EloGradingInput(BaseModel):
    instruction: str
    first_completion: str
    second_completion: str


class EloEvaluationLogic(
    IncrementalEvaluationLogic[Input, Output, ExpectedOutput, Matches]
):
    def __init__(self) -> None:
        super().__init__()
        self._previous_run_output_ids: list[set[str]] = []

    def set_previous_run_output_ids(
        self, previous_run_output_ids: list[set[str]]
    ) -> None:
        self._previous_run_output_ids = previous_run_output_ids

    def do_incremental_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        outputs: list[SuccessfulExampleOutput[Output]],
        already_evaluated_outputs: list[list[SuccessfulExampleOutput[Output]]],
    ) -> Matches:
        pairs = combinations(outputs, 2)
        unique_pre_evaluated_runs: set[str] = set()

        for pre_run_output in already_evaluated_outputs:
            for current_output in pre_run_output:
                unique_pre_evaluated_runs.add(current_output.run_id)

        return Matches(
            comparison_evaluations=[
                ComparisonEvaluation(
                    first_player=player_a.run_id,
                    second_player=player_b.run_id,
                    outcome=self.grade(player_a, player_b, example),
                )
                for [player_a, player_b] in pairs
                if unique_pre_evaluated_runs is None
                or len(unique_pre_evaluated_runs) == 0
                or not (
                    player_a.run_id in unique_pre_evaluated_runs
                    and player_b.run_id in unique_pre_evaluated_runs
                )
            ]
        )

    @abstractmethod
    def grade(
        self,
        first: SuccessfulExampleOutput[Output],
        second: SuccessfulExampleOutput[Output],
        example: Example[Input, ExpectedOutput],
    ) -> MatchOutcome:
        """Returns a :class: `MatchOutcome` for the provided two contestants on the given example.

        Defines the use case specific logic how to determine the winner of the two provided outputs.


        Args:
            first: Instance of :class: `SuccessfulExampleOutut[Output]` of the first contestant in the comparison
            second: Instance of :class: `SuccessfulExampleOutut[Output]` of the second contestant in the comparison
            example: Datapoint of :class: `Example` on which the two outputs were generated

        Returns:
            Instance of :class: `MatchOutcome`
        """
        pass
