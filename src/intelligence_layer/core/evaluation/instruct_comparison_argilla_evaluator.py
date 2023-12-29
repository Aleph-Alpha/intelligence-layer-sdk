from collections import defaultdict
from enum import Enum
from itertools import combinations
from random import choice, shuffle
from typing import Iterable, Mapping, Optional, Sequence, cast

from pydantic import BaseModel

from intelligence_layer.connectors import Field
from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaEvaluation,
    Question,
    RecordData,
)
from intelligence_layer.core.complete import InstructInput, PromptOutput
from intelligence_layer.core.evaluation.accumulator import MeanAccumulator
from intelligence_layer.core.evaluation.domain import Example, SuccessfulExampleOutput
from intelligence_layer.core.evaluation.evaluator import (
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    DatasetRepository,
)


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


class AggregatedInstructComparison(BaseModel):
    scores: Mapping[str, PlayerScore]


class InstructComparisonArgillaEvaluator(
    ArgillaEvaluator[
        InstructInput,
        PromptOutput,
        None,
        AggregatedInstructComparison,
    ]
):
    KEY_INSTRUCTION = "instruction"
    KEY_INPUT = "input"
    KEY_RESPONSE_1 = "first"
    KEY_RESPONSE_2 = "second"
    KEY_QUESTION = "winner"
    OPTIONS = [1, 2, 3]

    def __init__(
        self,
        evaluation_repository: ArgillaEvaluationRepository,
        dataset_repository: DatasetRepository,
        workspace_id: str,
        high_priority_runs: Optional[frozenset[str]] = None,
    ) -> None:
        fields = [
            Field(name=self.KEY_INSTRUCTION, title="Instruction"),
            Field(name=self.KEY_INPUT, title="Input"),
            Field(name=self.KEY_RESPONSE_1, title="Response 1"),
            Field(name=self.KEY_RESPONSE_2, title="Response 2"),
        ]
        questions = [
            Question(
                name=self.KEY_QUESTION,
                title="Which response is better?",
                description="1: The first completion is better.\n2: The second completion is better.\n3: They are both equally good.",
                options=self.OPTIONS,
            )
        ]

        super().__init__(
            evaluation_repository,
            dataset_repository,
            workspace_id,
            fields,
            questions,
        )
        self._high_priority_runs = high_priority_runs

    def _to_record(
        self,
        example: Example[InstructInput, None],
        *example_outputs: SuccessfulExampleOutput[PromptOutput],
    ) -> Sequence[RecordData]:
        def create_record_data(
            first: SuccessfulExampleOutput[PromptOutput],
            second: SuccessfulExampleOutput[PromptOutput],
        ) -> RecordData:
            if choice([True, False]):
                first, second = second, first
            return RecordData(
                content={
                    self.KEY_INSTRUCTION: example.input.instruction,
                    self.KEY_INPUT: example.input.input or "",
                    self.KEY_RESPONSE_1: first.output.completion,
                    self.KEY_RESPONSE_2: second.output.completion,
                },
                example_id=example.id,
                metadata={
                    self.KEY_RESPONSE_1: first.run_id,
                    self.KEY_RESPONSE_2: second.run_id,
                },
            )

        pairs = combinations(example_outputs, 2)
        return [
            create_record_data(first, second)
            for [first, second] in pairs
            if self._high_priority_runs is None
            or any(
                run_id in self._high_priority_runs
                for run_id in [first.run_id, second.run_id]
            )
        ]

    def aggregate(
        self, evaluations: Iterable[ArgillaEvaluation]
    ) -> AggregatedInstructComparison:
        def build_tournaments(
            evaluations: Iterable[ArgillaEvaluation],
        ) -> tuple[Mapping[str, Sequence[Payoff]], set[str]]:
            players: set[str] = set()
            # we group by example id to get a tournament round per example
            matches: dict[str, list[Payoff]] = defaultdict(list)
            for evaluation in evaluations:
                response = evaluation.responses[self.KEY_QUESTION]
                assert isinstance(response, int)
                matches[evaluation.example_id].append(
                    Payoff(
                        player1=evaluation.metadata[self.KEY_RESPONSE_1],
                        player2=evaluation.metadata[self.KEY_RESPONSE_2],
                        matrix=PayoffMatrix.from_rank_literal(response),
                    )
                )
                players.add(evaluation.metadata[self.KEY_RESPONSE_1])
                players.add(evaluation.metadata[self.KEY_RESPONSE_2])
            return cast(Mapping[str, Sequence[Payoff]], matches), players

        tournaments, players = build_tournaments(evaluations)

        accumulators = {p: MeanAccumulator() for p in players}
        tournaments_list = list(tournaments.items())
        for _ in range(100):
            elo_calc = EloCalculator(players)
            shuffle(tournaments_list)
            for _, tournament in tournaments_list:
                elo_calc.calculate_tournament(tournament)
            for p in players:
                accumulators[p].add(elo_calc.ratings[p])

        win_rate_calc = WinRateCalculator(players)
        win_rate = win_rate_calc.calculate(
            [battle for tournament in tournaments.values() for battle in tournament]
        )

        return AggregatedInstructComparison(
            scores={
                p: PlayerScore(elo=acc.extract(), win_rate=win_rate[p])
                for p, acc in accumulators.items()
            },
        )
