import random
from abc import ABC, abstractmethod
from collections import Counter
from itertools import combinations
from typing import Iterable, Mapping, Optional

from pydantic import BaseModel

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core import CompleteOutput, Input, InstructInput, Output
from intelligence_layer.evaluation import Aggregator
from intelligence_layer.evaluation.aggregation.accumulator import MeanAccumulator
from intelligence_layer.evaluation.aggregation.aggregation_repository import (
    AggregationRepository,
)
from intelligence_layer.evaluation.base_logic import AggregationLogic, EvaluationLogic
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.domain import (
    AggregatedEvaluation,
    Example,
    ExpectedOutput,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.elo import (
    EloCalculator,
    MatchOutcome,
    WinRateCalculator,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    ArgillaEvaluationRepository,
    EvaluationRepository,
    RecordDataSequence,
)
from intelligence_layer.evaluation.evaluation.evaluator import Evaluator
from intelligence_layer.evaluation.run.run_repository import RunRepository


class ArgillaEvaluationLogic(
    EvaluationLogic[Input, Output, ExpectedOutput, RecordDataSequence], ABC
):
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> RecordDataSequence:
        return self._to_record(example, *output)

    @abstractmethod
    def _to_record(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> RecordDataSequence:
        """This method is responsible for translating the `Example` and `Output` of the task to :class:`RecordData`


        Args:
            example: The example to be translated.
            output: The output of the example that was run.
        """
        ...


class ArgillaEvaluator(
    Evaluator[Input, Output, ExpectedOutput, ArgillaEvaluation],
    ABC,
):
    """Evaluator used to integrate with Argilla (https://github.com/argilla-io/argilla).

    Use this evaluator if you would like to easily do human eval.
    This evaluator runs a dataset and sends the input and output to Argilla to be evaluated.

     Arguments:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        description: Human-readable description for the evaluator.
        evaluation_logic: The logic to use for evaluation.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        ArgillaEvaluation: Interface of the metrics that come from the Argilla task`.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: ArgillaEvaluationRepository,
        description: str,
        evaluation_logic: ArgillaEvaluationLogic[Input, Output, ExpectedOutput],
    ) -> None:
        super().__init__(
            dataset_repository,
            run_repository,
            evaluation_repository,
            description,
            evaluation_logic,  # type: ignore
        )

    def evaluation_type(self) -> type[ArgillaEvaluation]:  # type: ignore
        return ArgillaEvaluation


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


class InstructComparisonArgillaEvaluationLogic(
    ArgillaEvaluationLogic[InstructInput, CompleteOutput, None]
):
    def __init__(
        self,
        workspace_id: str,
        fields: Mapping[str, Field],
        high_priority_runs: Optional[frozenset[str]] = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._fields = fields
        self._high_priority_runs = high_priority_runs

    def _to_record(
        self,
        example: Example[InstructInput, None],
        *outputs: SuccessfulExampleOutput[CompleteOutput],
    ) -> RecordDataSequence:
        pairs = combinations(outputs, 2)
        return RecordDataSequence(
            records=[
                self._create_record_data(example, first, second)
                for [first, second] in pairs
                if self._high_priority_runs is None
                or any(
                    run_id in self._high_priority_runs
                    for run_id in [first.run_id, second.run_id]
                )
            ]
        )

    def _create_record_data(
        self,
        example: Example[InstructInput, None],
        first: SuccessfulExampleOutput[CompleteOutput],
        second: SuccessfulExampleOutput[CompleteOutput],
    ) -> RecordData:
        if random.choice([True, False]):
            first, second = second, first
        return RecordData(
            content={
                self._fields["KEY_INSTRUCTION"].name: example.input.instruction,
                self._fields["KEY_INPUT"].name: example.input.input or "",
                self._fields["KEY_RESPONSE_1"].name: first.output.completion,
                self._fields["KEY_RESPONSE_2"].name: second.output.completion,
            },
            example_id=example.id,
            metadata={
                self._fields["KEY_RESPONSE_1"].name: first.run_id,
                self._fields["KEY_RESPONSE_2"].name: second.run_id,
            },
        )


def create_instruct_comparison_argilla_evaluation_classes(
    workspace_id: str,
    evaluation_repository: EvaluationRepository,
    argilla_client: ArgillaClient,
    high_priority_runs: Optional[frozenset[str]] = None,
) -> tuple[InstructComparisonArgillaEvaluationLogic, ArgillaEvaluationRepository]:
    KEY_INSTRUCTION = "instruction"
    KEY_INPUT = "input"
    KEY_RESPONSE_1 = "first"
    KEY_RESPONSE_2 = "second"
    KEY_QUESTION = "winner"
    OPTIONS = [1, 2, 3]

    fields = {
        "KEY_INSTRUCTION": Field(name=KEY_INSTRUCTION, title="Instruction"),
        "KEY_INPUT": Field(name=KEY_INPUT, title="Input"),
        "KEY_RESPONSE_1": Field(name=KEY_RESPONSE_1, title="Response 1"),
        "KEY_RESPONSE_2": Field(name=KEY_RESPONSE_2, title="Response 2"),
    }
    questions = [
        Question(
            name=KEY_QUESTION,
            title="Which response is better?",
            description="1: The first completion is better.\n2: The second completion is better.\n3: They are both equally good.",
            options=OPTIONS,
        )
    ]

    return InstructComparisonArgillaEvaluationLogic(
        workspace_id, fields, high_priority_runs
    ), ArgillaEvaluationRepository(
        evaluation_repository,
        argilla_client,
        workspace_id,
        list(fields.values()),
        questions,
    )
