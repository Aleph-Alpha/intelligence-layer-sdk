from itertools import combinations
from typing import Sequence

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.evaluation import EloCalculator, MatchOutcome, WinRateCalculator


@fixture
def players() -> Sequence[str]:
    return [str(i + 1) for i in range(10)]


@fixture
def matches(players: Sequence[str]) -> Sequence[tuple[str, str, MatchOutcome]]:
    return [
        (player_a, player_b, MatchOutcome.A_WINS)
        for player_a, player_b in combinations(players, 2)
    ]


class MatchOutcomeModel(BaseModel):
    match_outcome: MatchOutcome


def test_match_outcome_serializes() -> None:
    match_outcome_model = MatchOutcomeModel(match_outcome=MatchOutcome.A_WINS)
    dumped = match_outcome_model.model_dump_json()
    loaded = MatchOutcomeModel.model_validate_json(dumped)

    assert loaded == match_outcome_model


def test_elo_calculator_works(
    players: Sequence[str], matches: Sequence[tuple[str, str, MatchOutcome]]
) -> None:
    elo_calculator = EloCalculator(players)
    elo_calculator.calculate(matches)

    sorted_scores = {
        k: v
        for k, v in sorted(
            elo_calculator.ratings.items(), key=lambda item: item[1], reverse=True
        )
    }
    assert [int(i) for i in players] == [int(i) for i in sorted_scores.keys()]
    assert (
        round(sum(score for score in sorted_scores.values()) / len(sorted_scores), 0)
        == 1500
    )


def test_win_rate_calculator_works(
    players: Sequence[str], matches: Sequence[tuple[str, str, MatchOutcome]]
) -> None:
    win_rate_calculator = WinRateCalculator(players)
    scores = win_rate_calculator.calculate(matches)

    sorted_scores = {
        k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    }
    assert [int(i) for i in players] == [int(i) for i in sorted_scores.keys()]
    assert (
        round(
            sum(score for score in sorted_scores.values()) / len(sorted_scores),
            5,
        )
        == 0.5
    )
