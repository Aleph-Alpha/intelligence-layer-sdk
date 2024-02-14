import random
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors import Field
from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaEvaluation,
    Question,
    RecordData,
)
from intelligence_layer.core.complete import InstructInput, PromptOutput
from intelligence_layer.evaluation import (
    Example,
    MeanAccumulator,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.elo import (
    AutomatedEloComparison,
    EloCalculator,
    EloComparison,
    PlayerScore,
    WinRateCalculator,
    build_tournaments,
)
from intelligence_layer.evaluation.evaluator import ArgillaEvaluator


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
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: ArgillaEvaluationRepository,
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
            dataset_repository,
            run_repository,
            evaluation_repository,
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
        elo_evaluations = [
            AutomatedEloComparison(
                outputs=[
                    EloComparison(
                        example_id=evaluation.example_id,
                        winner=int(evaluation.responses["winner"]),
                        first_run_id=evaluation.metadata["first"],
                        second_run_id=evaluation.metadata["second"],
                    )
                ]
            )
            for evaluation in evaluations
        ]
        tournaments, players = build_tournaments(elo_evaluations)

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
