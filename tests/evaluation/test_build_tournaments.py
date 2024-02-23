from itertools import combinations
from typing import Sequence

import pytest

from intelligence_layer.evaluation.elo import (
    AutomatedEloComparison,
    EloComparison,
    PayoffMatrix,
    build_tournaments,
)


@pytest.mark.parametrize(
    "comparisons",
    [[], [AutomatedEloComparison(outputs=[])]],
)
def test_build_tournaments_returns_empty_data_for_no_comparisons(
    comparisons: Sequence[AutomatedEloComparison],
) -> None:
    matches, players = build_tournaments(comparisons)

    assert len(players) == 0
    assert len(matches) == 0


@pytest.mark.parametrize(
    "winner, expected_winner",
    [
        (1, PayoffMatrix.PLAYER_1_WINS),
        (2, PayoffMatrix.PLAYER_2_WINS),
        (3, PayoffMatrix.DRAW),
    ],
)
def test_build_tournaments_returns_correct_data_for_multiple_comparisons(
    winner: int, expected_winner: PayoffMatrix
) -> None:
    example_ids = ["example_id_1", "example_id_2"]
    run_ids_1 = ["run_id_1", "run_id_2", "run_id_3"]
    run_ids_2 = ["run_id_4", "run_id_5"]
    example_1_comparisons = [
        EloComparison(
            example_id=example_ids[0],
            winner=winner,
            first_run_id=run_id_1,
            second_run_id=run_id_2,
        )
        for run_id_1, run_id_2 in combinations(run_ids_1, 2)
    ]
    example_2_comparisons = [
        EloComparison(
            example_id=example_ids[1],
            winner=winner,
            first_run_id=run_id_1,
            second_run_id=run_id_2,
        )
        for run_id_1, run_id_2 in combinations(run_ids_2, 2)
    ]
    comparisons = [
        AutomatedEloComparison(outputs=example_1_comparisons + example_2_comparisons)
    ]

    matches, players = build_tournaments(comparisons)

    # number of players is number of unique run IDs
    assert players == set(run_ids_1 + run_ids_2)
    # matches contain given example IDs
    assert set(matches.keys()) == set(example_ids)
    # the number of matches is the number of given comparisons
    assert len(matches[example_ids[0]]) == len(example_1_comparisons)
    assert len(matches[example_ids[1]]) == len(example_2_comparisons)
    # the payoffs have the correct players and winner
    for example_id, example_comparisons in zip(
        example_ids, [example_1_comparisons, example_2_comparisons]
    ):
        for i, comparison in enumerate(example_comparisons):
            assert matches[example_id][i].player1 == comparison.first_run_id
            assert matches[example_id][i].player2 == comparison.second_run_id
            assert matches[example_id][i].matrix == expected_winner
