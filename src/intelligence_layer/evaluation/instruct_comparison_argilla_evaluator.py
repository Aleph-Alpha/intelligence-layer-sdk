import random
from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence, cast

from pydantic import BaseModel

from intelligence_layer.connectors import Field
from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaEvaluation,
    Question,
    RecordData,
)
from intelligence_layer.core.complete import InstructInput, PromptOutput
from intelligence_layer.evaluation.accumulator import MeanAccumulator
from intelligence_layer.evaluation.domain import Example, SuccessfulExampleOutput
from intelligence_layer.evaluation.elo import (
    EloCalculator,
    Payoff,
    PayoffMatrix,
    PlayerScore,
    WinRateCalculator,
)
from intelligence_layer.evaluation.evaluator import (
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    DatasetRepository,
)


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
        description: str,
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
            description,
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
            if random.choice([True, False]):
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
            random.shuffle(tournaments_list)
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
