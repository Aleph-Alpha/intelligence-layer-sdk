from abc import abstractmethod
from enum import Enum
from itertools import combinations
from typing import Sequence, final

from pydantic import BaseModel

from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import EvaluationLogic
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


class EloEvaluationLogic(EvaluationLogic[Input, Output, ExpectedOutput, Matches]):
    @final
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> Matches:
        pairs = combinations(output, 2)
        return Matches(
            comparison_evaluations=[
                ComparisonEvaluation(
                    first_player=player_a.run_id,
                    second_player=player_b.run_id,
                    outcome=self.grade(player_a, player_b, example),
                )
                for [player_a, player_b] in pairs
            ]
        )

    @abstractmethod
    def grade(
        self,
        output_a: SuccessfulExampleOutput[Output],
        output_b: SuccessfulExampleOutput[Output],
        example: Example[Input, ExpectedOutput],
    ) -> MatchOutcome:
        pass
