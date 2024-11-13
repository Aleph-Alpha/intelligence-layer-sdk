import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from intelligence_layer.evaluation.aggregation.accumulator import MeanAccumulator
from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic
from intelligence_layer.evaluation.evaluation.evaluator.incremental_evaluator import (
    ComparisonEvaluation,
    Matches,
    MatchOutcome,
)


class PlayerScore(BaseModel):
    elo: float
    elo_standard_error: float
    win_rate: float
    num_matches: int


class AggregatedComparison(BaseModel):
    scores: Mapping[str, PlayerScore]


class EloAggregationAdapter:
    @staticmethod
    def aggregate(evaluations: Iterable[ComparisonEvaluation]) -> AggregatedComparison:
        evaluations = list(evaluations)
        player_counter = Counter(
            player
            for comparison_evaluation in evaluations
            for player in [
                comparison_evaluation.first_player,
                comparison_evaluation.second_player,
            ]
        )

        player_counts = dict(player_counter)
        players = player_counts.keys()

        accumulators = {p: MeanAccumulator() for p in players}
        for _ in range(100):
            elo_calc = EloCalculator(players)
            random.shuffle(evaluations)
            elo_calc.calculate(evaluations)
            for p in players:
                accumulators[p].add(elo_calc.ratings[p])

        win_rate_calc = WinRateCalculator(players)
        win_rate = win_rate_calc.calculate(evaluations)

        return AggregatedComparison(
            scores={
                p: PlayerScore(
                    elo=acc.extract(),
                    elo_standard_error=acc.standard_error(),
                    win_rate=win_rate[p],
                    num_matches=player_counts[p],
                )
                for p, acc in accumulators.items()
            },
        )


class EloCalculator:
    def __init__(
        self,
        players: Iterable[str],
        k_start: float = 20.0,
        k_floor: float = 10.0,
        decay_factor: float = 0.0005,
    ) -> None:
        self.ratings: dict[str, float] = {player: 1500.0 for player in players}
        self._match_counts: dict[str, int] = defaultdict(int)
        self._k_ceiling = k_start - k_floor
        self._k_floor = k_floor
        self._decay_factor = decay_factor

    def _calc_k_factor(self, player: str) -> float:
        n = self._match_counts.get(player) or 0
        return self._k_ceiling * np.exp(-self._decay_factor * n) + self._k_floor

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

    def calculate(self, matches: Sequence[ComparisonEvaluation]) -> None:
        for match in matches:
            dif_a, dif_b = self._calc_difs(
                match.outcome, match.first_player, match.second_player
            )
            self.ratings[match.first_player] += dif_a
            self.ratings[match.second_player] += dif_b
            self._match_counts[match.first_player] += 1
            self._match_counts[match.second_player] += 1


class WinRateCalculator:
    def __init__(self, players: Iterable[str]) -> None:
        self.match_count: dict[str, int] = {p: 0 for p in players}
        self.win_count: dict[str, float] = {p: 0 for p in players}

    def calculate(self, matches: Sequence[ComparisonEvaluation]) -> Mapping[str, float]:
        for match in matches:
            self.match_count[match.first_player] += 1
            self.match_count[match.second_player] += 1
            self.win_count[match.first_player] += match.outcome.payoff[0]
            self.win_count[match.second_player] += match.outcome.payoff[1]

        return {
            player: self.win_count[player] / match_count
            for player, match_count in self.match_count.items()
        }


class ComparisonEvaluationAggregationLogic(
    AggregationLogic[ComparisonEvaluation, AggregatedComparison]
):
    def aggregate(
        self, evaluations: Iterable[ComparisonEvaluation]
    ) -> AggregatedComparison:
        return EloAggregationAdapter.aggregate(evaluations)


class MatchesAggregationLogic(AggregationLogic[Matches, AggregatedComparison]):
    def aggregate(self, evaluations: Iterable[Matches]) -> AggregatedComparison:
        flattened_matches = [
            comparison_evaluation
            for match in evaluations
            for comparison_evaluation in match.comparison_evaluations
        ]
        return EloAggregationAdapter.aggregate(flattened_matches)
