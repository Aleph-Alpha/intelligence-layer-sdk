from collections import defaultdict
from itertools import combinations
from typing import Iterable, Sequence
from uuid import uuid4

from aleph_alpha_client import CompletionResponse
from aleph_alpha_client.completion import CompletionResult
from faker import Faker
from pytest import fixture

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core.complete import InstructInput, PromptOutput
from intelligence_layer.core.prompt_template import RichPrompt
from intelligence_layer.core.tracer import utc_now
from intelligence_layer.evaluation import (
    ArgillaEvaluator,
    EloCalculator,
    Example,
    ExampleOutput,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    InstructComparisonArgillaAggregationLogic,
    Payoff,
    PayoffMatrix,
    RunOverview,
)
from intelligence_layer.evaluation.argilla import (
    AggregatedInstructComparison,
    create_instruct_comparison_argilla_evaluation_classes,
)
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    InMemoryAggregationRepository,
)


class ArgillaFake(ArgillaClient):
    def __init__(self) -> None:
        self.records: dict[str, list[RecordData]] = defaultdict(list)

    def create_dataset(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        return str(uuid4())

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        self.records[dataset_id].append(record)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        return [
            ArgillaEvaluation(
                example_id=r.example_id,
                record_id=str(uuid4()),
                responses={
                    "winner": (
                        1 if int(r.metadata["first"]) < int(r.metadata["second"]) else 2
                    )
                },
                metadata=r.metadata,
            )
            for r in self.records[dataset_id]
        ]

    def record_data(self, dataset_id: str) -> Sequence[RecordData]:
        return self.records.get(dataset_id, [])


@fixture
def argilla_fake() -> ArgillaClient:
    return ArgillaFake()


@fixture
def evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    argilla_fake: ArgillaClient,
    argilla_aggregation_logic: InstructComparisonArgillaAggregationLogic,
) -> ArgillaEvaluator[InstructInput, PromptOutput, None, AggregatedInstructComparison]:
    (
        evaluation_logic,
        evaluation_repository,
    ) = create_instruct_comparison_argilla_evaluation_classes(
        "workspace", in_memory_evaluation_repository, argilla_fake, None
    )
    return ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        evaluation_repository,
        in_memory_aggregation_repository,
        "instruct-evaluator",
        evaluation_logic,
        argilla_aggregation_logic,
    )


@fixture
def any_instruct_output() -> PromptOutput:
    faker = Faker()
    return PromptOutput(
        response=CompletionResponse(
            model_version="",
            completions=[CompletionResult(completion=faker.text())],
            num_tokens_generated=0,
            num_tokens_prompt_total=0,
        ),
        rich_prompt=RichPrompt(items=[], ranges={}),
    )


def test_evaluate_run_submits_pairwise_comparison_records(
    evaluator: ArgillaEvaluator[
        InstructInput,
        PromptOutput,
        None,
        AggregatedInstructComparison,
    ],
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    any_instruct_output: PromptOutput,
    argilla_fake: ArgillaFake,
) -> None:
    run_count = 10
    run_ids = [f"{i}" for i in range(run_count)]
    example_id = "example_id"
    instruction = "inst"
    instruction_input = "some text"
    dataset_id = in_memory_dataset_repository.create_dataset(
        [
            Example(
                id=example_id,
                input=InstructInput(instruction=instruction, input=instruction_input),
                expected_output=None,
            )
        ]
    )
    for run_id in run_ids:
        in_memory_run_repository.store_example_output(
            example_output=ExampleOutput(
                run_id=run_id, example_id="example_id", output=any_instruct_output
            )
        )
        in_memory_run_repository.store_run_overview(
            RunOverview(
                dataset_id=dataset_id,
                id=run_id,
                start=utc_now(),
                end=utc_now(),
                failed_example_count=0,
                successful_example_count=0,
                description="runner",
            )
        )

    evaluation_overview = evaluator.evaluate_runs(*run_ids)

    pairs = combinations(run_ids, 2)
    assert sorted(
        tuple(sorted((record_data.metadata["first"], record_data.metadata["second"])))
        for record_data in argilla_fake.record_data(evaluation_overview.id)
    ) == sorted(pairs)

    elo_score = evaluator.aggregate_evaluation(evaluation_overview.id)
    scores = elo_score.statistics.scores
    # lower id always wins, should be sorted
    for i in range(run_count - 1):
        assert scores[run_ids[i]].elo > scores[run_ids[i + 1]].elo
        assert scores[run_ids[i]].win_rate > scores[run_ids[i + 1]].win_rate


def test_evaluate_run_only_evaluates_high_priority(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    any_instruct_output: PromptOutput,
    argilla_aggregation_logic: InstructComparisonArgillaAggregationLogic,
    argilla_fake: ArgillaFake,
) -> None:
    relevant_ids = frozenset({"1", "2"})
    eval_logic, eval_repository = create_instruct_comparison_argilla_evaluation_classes(
        "workspace", in_memory_evaluation_repository, argilla_fake, relevant_ids
    )
    evaluator = ArgillaEvaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        eval_repository,
        in_memory_aggregation_repository,
        "instruct-evaluator",
        eval_logic,
        argilla_aggregation_logic,
    )

    run_count = 10
    run_ids = [f"{i}" for i in range(run_count)]
    example_id = "example_id"
    instruction = "inst"
    instruction_input = "some text"
    dataset_id = in_memory_dataset_repository.create_dataset(
        [
            Example(
                id=example_id,
                input=InstructInput(instruction=instruction, input=instruction_input),
                expected_output=None,
            )
        ]
    )
    for run_id in run_ids:
        in_memory_run_repository.store_example_output(
            example_output=ExampleOutput(
                run_id=run_id, example_id="example_id", output=any_instruct_output
            )
        )
        in_memory_run_repository.store_run_overview(
            RunOverview(
                dataset_id=dataset_id,
                id=run_id,
                start=utc_now(),
                end=utc_now(),
                failed_example_count=0,
                successful_example_count=0,
                description="runner",
            )
        )

    evaluation_overview = evaluator.evaluate_runs(*run_ids)

    def relevant_ids_in_record(record: RecordData) -> bool:
        players = [record.metadata["first"], record.metadata["second"]]
        return any(id in players for id in relevant_ids)

    records = argilla_fake.record_data(evaluation_overview.id)
    assert all(relevant_ids_in_record(record) for record in records)
    assert len(records) == sum(run_count - (i + 1) for i in range(len(relevant_ids)))


def test_elo_calculating_works_as_expected() -> None:
    player1 = "player1"
    player2 = "player2"
    matches = [
        Payoff(
            player1=player1,
            player2=player2,
            matrix=PayoffMatrix.PLAYER_1_WINS,
        )
        for i in range(10)
    ]
    elo = EloCalculator([player1, player2])
    elo.calculate_tournament(matches)

    assert elo.ratings[player1] == 1600
    assert elo.ratings[player2] == 1400

    comeback_matches = [
        Payoff(
            player1=player1,
            player2=player2,
            matrix=PayoffMatrix.PLAYER_2_WINS,
        )
        for i in range(10)
    ]
    elo.calculate_tournament(comeback_matches)

    assert elo.ratings[player2] > elo.ratings[player1]
