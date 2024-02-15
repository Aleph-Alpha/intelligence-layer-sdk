from collections import defaultdict
from enum import Enum
from typing import Iterable, Mapping, Sequence, cast

from pydantic import BaseModel


class PayoffMatrix(Enum):
    PLAYER_1_WINS = (1, 0)
    DRAW = (0.5, 0.5)
    PLAYER_2_WINS = (0, 1)

    @staticmethod
    def from_rank_literal(rank: int) -> "PayoffMatrix":
        match rank:
            case 1:
                return PayoffMatrix.PLAYER_1_WINS
            case 2:
                return PayoffMatrix.PLAYER_2_WINS
            case 3:
                return PayoffMatrix.DRAW
            case _:
                raise ValueError(f"Got unexpected rank {rank}")


class Payoff(BaseModel):
    player1: str
    player2: str
    matrix: PayoffMatrix


class EloCalculator:
    def __init__(self, players: Iterable[str], k_factor: int = 20) -> None:
        self.ratings: dict[str, float] = {p: 1500 for p in players}
        self._k_factor = k_factor

    @staticmethod
    def _update_dict_keys(
        to_be_updated: dict[str, float], update_with: Mapping[str, float]
    ) -> None:
        for key, val in update_with.items():
            to_be_updated[key] += val

    def calculate_tournament(self, payoffs: Sequence[Payoff]) -> None:
        tournament_difs = {p: 0.0 for p in self.ratings.keys()}

        for payoff in payoffs:
            dif_map = self._get_difs(payoff)
            self._update_dict_keys(tournament_difs, dif_map)

        self._update_dict_keys(self.ratings, tournament_difs)

    def _calc_expected_win_rates(
        self, player_a: str, player_b: str
    ) -> tuple[float, float]:
        rating_a, rating_b = self.ratings[player_a], self.ratings[player_b]
        expected_win_rate_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return expected_win_rate_a, 1 - expected_win_rate_a

    def _get_difs(self, preference_result: Payoff) -> Mapping[str, float]:
        expected_win_rate_a, expected_win_rate_b = self._calc_expected_win_rates(
            preference_result.player1, preference_result.player2
        )
        dif_a, dif_b = self._calc_difs(
            preference_result.matrix,
            expected_win_rate_a,
            expected_win_rate_b,
        )
        return {preference_result.player1: dif_a, preference_result.player2: dif_b}

    def _calc_difs(
        self,
        payoff: PayoffMatrix,
        expected_win_rate_a: float,
        expected_win_rate_b: float,
    ) -> tuple[float, float]:
        def calc_dif(actual: float, expected_win_rate: float) -> float:
            return self._k_factor * (actual - expected_win_rate)

        actual_a, actual_b = payoff.value
        dif_a = calc_dif(actual_a, expected_win_rate_a)
        dif_b = calc_dif(actual_b, expected_win_rate_b)
        return dif_a, dif_b


class WinRateCalculator:
    def __init__(self, players: Iterable[str]) -> None:
        self.match_count: dict[str, int] = {p: 0 for p in players}
        self.win_count: dict[str, float] = {p: 0 for p in players}

    def calculate(self, payoffs: Sequence[Payoff]) -> Mapping[str, float]:
        for result in payoffs:
            self.match_count[result.player1] += 1
            self.match_count[result.player2] += 1
            self.win_count[result.player1] += result.matrix.value[0]
            self.win_count[result.player2] += result.matrix.value[1]

        return {
            player: self.win_count[player] / match_count
            for player, match_count in self.match_count.items()
        }


class PlayerScore(BaseModel):
    elo: float
    win_rate: float


class EloComparison(BaseModel):
    example_id: str
    winner: int
    first_run_id: str
    second_run_id: str


class AutomatedEloComparison(BaseModel):
    outputs: Sequence[EloComparison]


def build_tournaments(
    evaluations: Iterable[AutomatedEloComparison],
) -> tuple[Mapping[str, Sequence[Payoff]], set[str]]:
    players: set[str] = set()
    # we group by example id to get a tournament round per example
    matches: dict[str, list[Payoff]] = defaultdict(list)
    for instruct_comparison in evaluations:
        for evaluation in instruct_comparison.outputs:
            winner = evaluation.winner
            assert isinstance(winner, int)
            matches[evaluation.example_id].append(
                Payoff(
                    player1=evaluation.first_run_id,
                    player2=evaluation.second_run_id,
                    matrix=PayoffMatrix.from_rank_literal(winner),
                )
            )
            players.add(evaluation.first_run_id)
            players.add(evaluation.second_run_id)
    return cast(Mapping[str, Sequence[Payoff]], matches), players
