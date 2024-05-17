from abc import abstractmethod
from enum import Enum
from itertools import combinations
from typing import Sequence, final

from pydantic import BaseModel

from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.evaluator.incremental_evaluator import (
    IncrementalEvaluationLogic,
)
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput


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

    @final
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
        """Returns a :class: `MatchOutcome`for the provided two contestants on the given example.
        Defines the use case specific logic how to determine the winner of the two provided outputs.


        Args:
            first: Instance of :class: `SuccessfulExampleOutut[Output]` of the first contestant in the comparison
            second: Instance of :class: `SuccessfulExampleOutut[Output]` of the second contestant in the comparison
            example: Datapoint of :class: `Example` on which the two outputs were generated

        Return:
            Instance of :class: `MatchOutcome`
        """
        pass
