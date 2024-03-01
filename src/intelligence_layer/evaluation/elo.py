from enum import Enum
from typing import Iterable, Mapping, Sequence

from pydantic import BaseModel


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


class Match(BaseModel):
    player_a: str
    player_b: str
    outcome: MatchOutcome


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

    def calculate(self, matches: Sequence[Match]) -> None:
        difs = {p: 0.0 for p in self.ratings.keys()}

        for match_ in matches:
            dif_map = self._get_difs(match_)
            self._update_dict_keys(difs, dif_map)

        self._update_dict_keys(self.ratings, difs)

    def _calc_expected_win_rates(
        self, player_a: str, player_b: str
    ) -> tuple[float, float]:
        rating_a, rating_b = self.ratings[player_a], self.ratings[player_b]
        expected_win_rate_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return expected_win_rate_a, 1 - expected_win_rate_a

    def _get_difs(self, preference_result: Match) -> Mapping[str, float]:
        expected_win_rate_a, expected_win_rate_b = self._calc_expected_win_rates(
            preference_result.player_a, preference_result.player_b
        )
        dif_a, dif_b = self._calc_difs(
            preference_result.outcome,
            expected_win_rate_a,
            expected_win_rate_b,
        )
        return {preference_result.player_a: dif_a, preference_result.player_b: dif_b}

    def _calc_difs(
        self,
        match_outcome: MatchOutcome,
        expected_win_rate_a: float,
        expected_win_rate_b: float,
    ) -> tuple[float, float]:
        def calc_dif(actual: float, expected_win_rate: float) -> float:
            return self._k_factor * (actual - expected_win_rate)

        actual_a, actual_b = match_outcome.payoff
        dif_a = calc_dif(actual_a, expected_win_rate_a)
        dif_b = calc_dif(actual_b, expected_win_rate_b)
        return dif_a, dif_b


class WinRateCalculator:
    def __init__(self, players: Iterable[str]) -> None:
        self.match_count: dict[str, int] = {p: 0 for p in players}
        self.win_count: dict[str, float] = {p: 0 for p in players}

    def calculate(self, matches: Sequence[Match]) -> Mapping[str, float]:
        for match_ in matches:
            self.match_count[match_.player_a] += 1
            self.match_count[match_.player_b] += 1
            self.win_count[match_.player_a] += match_.outcome.payoff[0]
            self.win_count[match_.player_b] += match_.outcome.payoff[1]

        return {
            player: self.win_count[player] / match_count
            for player, match_count in self.match_count.items()
        }
