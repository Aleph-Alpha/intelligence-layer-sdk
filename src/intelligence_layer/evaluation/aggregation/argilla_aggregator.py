import random
from abc import ABC
from collections import Counter
from typing import Iterable, Mapping

from pydantic import BaseModel

from intelligence_layer.connectors.argilla.argilla_client import ArgillaEvaluation
from intelligence_layer.evaluation.aggregation.accumulator import MeanAccumulator
from intelligence_layer.evaluation.aggregation.aggregation_repository import (
    AggregationRepository,
)
from intelligence_layer.evaluation.aggregation.aggregator import (
    AggregationLogic,
    Aggregator,
)
from intelligence_layer.evaluation.aggregation.domain import AggregatedEvaluation
from intelligence_layer.evaluation.elo import (
    EloCalculator,
    MatchOutcome,
    WinRateCalculator,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    ArgillaEvaluationRepository,
)


class ArgillaAggregator(
    Aggregator[ArgillaEvaluation, AggregatedEvaluation],
    ABC,
):
    """Aggregator used to aggregate Argilla (https://github.com/argilla-io/argilla) evaluations.

    You can fetch the results by using the `aggregate_evaluation` method.

     Arguments:
        evaluation_repository: The repository that will be used to store evaluation results.
        aggregation_repository: The repository that will be used to store aggregation results.
        description: Human-readable description for the evaluator.
        aggregation_logic: The logic to aggregate the evaluations.

    Generics:
        ArgillaEvaluation: Interface of the metrics that come from the Argilla task`.
        AggregatedEvaluation: The aggregated results of an evaluation run with a :class:`Dataset`.
    """

    def evaluation_type(self) -> type[ArgillaEvaluation]:  # type: ignore
        return ArgillaEvaluation

    def __init__(
        self,
        evaluation_repository: ArgillaEvaluationRepository,
        aggregation_repository: AggregationRepository,
        description: str,
        aggregation_logic: AggregationLogic[ArgillaEvaluation, AggregatedEvaluation],
    ) -> None:
        super().__init__(
            evaluation_repository,
            aggregation_repository,
            description,
            aggregation_logic,
        )


class PlayerScore(BaseModel):
    elo: float
    elo_standard_error: float
    win_rate: float
    num_matches: int


class AggregatedInstructComparison(BaseModel):
    scores: Mapping[str, PlayerScore]


class InstructComparisonArgillaAggregationLogic(
    AggregationLogic[ArgillaEvaluation, AggregatedInstructComparison]
):
    def aggregate(
        self, evaluations: Iterable[ArgillaEvaluation]
    ) -> AggregatedInstructComparison:
        flattened_evaluations = [
            (
                evaluation.metadata["first"],
                evaluation.metadata["second"],
                MatchOutcome.from_rank_literal(int(evaluation.responses["winner"])),
            )
            for evaluation in evaluations
        ]
        player_counter = Counter(
            player for match in flattened_evaluations for player in [match[0], match[1]]
        )
        player_counts = dict(player_counter)
        players = player_counts.keys()

        accumulators = {p: MeanAccumulator() for p in players}
        for _ in range(100):
            elo_calc = EloCalculator(players)
            random.shuffle(flattened_evaluations)
            elo_calc.calculate(flattened_evaluations)
            for p in players:
                accumulators[p].add(elo_calc.ratings[p])

        win_rate_calc = WinRateCalculator(players)
        win_rate = win_rate_calc.calculate(flattened_evaluations)

        return AggregatedInstructComparison(
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
