from collections import defaultdict
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np
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
    def __init__(
        self,
        players: Iterable[str],
        k_start: float = 20.0,
        k_floor: float = 5.0,
        decay_factor: float = 0.0005,
    ) -> None:
        self.ratings: dict[str, float] = {player: 1500.0 for player in players}
        self._match_counts: dict[str, int] = defaultdict(int)
        self._k_ceiling = k_start - k_floor
        self._k_floor = k_floor
        self._decay_factor = decay_factor

    def _calc_k_factor(self, player: str) -> float:
        n = self._match_counts.get(player) or 0
        # Mypy thinks this is Any
        return self._k_ceiling * np.exp(-self._decay_factor * n) + self._k_floor  # type: ignore

    def _calc_expected_win_rates(
        self, player_a: str, player_b: str
    ) -> tuple[float, float]:
        rating_a, rating_b = self.ratings[player_a], self.ratings[player_b]
        exp_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return exp_a, 1 - exp_a

    def _calc_difs(
        self, match_outcome: MatchOutcome, player_a: str, player_b: str
    ) -> tuple[float, float]:
        expected_win_rate_a, expected_win_rate_b = self._calc_expected_win_rates(
            player_a, player_b
        )
        actual_a, actual_b = match_outcome.payoff
        k_a, k_b = self._calc_k_factor(player_a), self._calc_k_factor(player_b)
        return k_a * (actual_a - expected_win_rate_a), k_b * (
            actual_b - expected_win_rate_b
        )

    def calculate(self, matches: Sequence[Match]) -> None:
        for m in matches:
            dif_a, dif_b = self._calc_difs(m.outcome, m.player_a, m.player_b)
            self.ratings[m.player_a] += dif_a
            self.ratings[m.player_b] += dif_b
            self._match_counts[m.player_a] += 1
            self._match_counts[m.player_b] += 1


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
