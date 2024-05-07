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
from intelligence_layer.core import CompleteOutput, InstructInput, utc_now
from intelligence_layer.evaluation import (
    AggregatedInstructComparison,
    ArgillaAggregator,
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    EloCalculator,
    Example,
    ExampleOutput,
    InMemoryAggregationRepository,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    InstructComparisonArgillaAggregationLogic,
    MatchOutcome,
    RunOverview,
)
from intelligence_layer.evaluation.evaluation.argilla_evaluator import (
    InstructComparisonArgillaEvaluation,
    InstructComparisonArgillaEvaluationLogic,
)


class ArgillaFake(ArgillaClient):
    def __init__(self) -> None:
        self.records: dict[str, list[RecordData]] = defaultdict(list)

    def ensure_dataset_exists(
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

    def split_dataset(self, dataset_id: str, n_splits: int) -> None:
        raise NotImplementedError


@fixture
def argilla_fake() -> ArgillaClient:
    return ArgillaFake()


@fixture
def evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    argilla_fake: ArgillaClient,
) -> ArgillaEvaluator[
    InstructInput, CompleteOutput, None, InstructComparisonArgillaEvaluation
]:
    evaluation_logic = InstructComparisonArgillaEvaluationLogic()

    return ArgillaEvaluator(
        dataset_repository=in_memory_dataset_repository,
        run_repository=in_memory_run_repository,
        evaluation_repository=in_memory_evaluation_repository,
        description="instruct-evaluator",
        workspace_id="workspace",
        argilla_client=argilla_fake,
        evaluation_logic=evaluation_logic,
    )


@fixture
def aggregator(
    argilla_repository: ArgillaEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    argilla_aggregation_logic: InstructComparisonArgillaAggregationLogic,
) -> ArgillaAggregator[AggregatedInstructComparison]:
    return ArgillaAggregator(
        argilla_repository,
        in_memory_aggregation_repository,
        "instruct-evaluator",
        argilla_aggregation_logic,
    )


@fixture
def any_instruct_output() -> CompleteOutput:
    faker = Faker()
    return CompleteOutput.from_completion_response(
        CompletionResponse(
            model_version="",
            completions=[CompletionResult(completion=faker.text())],
            num_tokens_generated=0,
            num_tokens_prompt_total=0,
        ),
    )


def create_dummy_dataset(
    in_memory_dataset_repository: InMemoryDatasetRepository,
) -> str:
    example_id = "example_id"
    instruction = "inst"
    instruction_input = "some text"

    return in_memory_dataset_repository.create_dataset(
        examples=[
            Example(
                id=example_id,
                input=InstructInput(instruction=instruction, input=instruction_input),
                expected_output=None,
            )
        ],
        dataset_name="test-dataset",
    ).id


def create_dummy_runs(
    in_memory_run_repository: InMemoryRunRepository,
    any_instruct_output: CompleteOutput,
    run_ids: Sequence[str],
    dataset_id: str,
) -> None:
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
                successful_example_count=1,
                description="runner",
            )
        )


def test_evaluate_run_submits_pairwise_comparison_records(
    evaluator: ArgillaEvaluator[
        InstructInput, CompleteOutput, None, InstructComparisonArgillaEvaluation
    ],
    aggregator: ArgillaAggregator[AggregatedInstructComparison],
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    any_instruct_output: CompleteOutput,
    argilla_fake: ArgillaFake,
) -> None:
    run_count = 10
    run_ids = [f"{i}" for i in range(run_count)]
    dataset_id = create_dummy_dataset(in_memory_dataset_repository)
    create_dummy_runs(
        in_memory_run_repository, any_instruct_output, run_ids, dataset_id
    )

    evaluation_overview = evaluator.evaluate_runs(*run_ids)

    pairs = combinations(run_ids, 2)
    assert sorted(
        tuple(sorted((record_data.metadata["first"], record_data.metadata["second"])))
        for record_data in argilla_fake.record_data(evaluation_overview.id)
    ) == sorted(pairs)

    elo_score = aggregator.aggregate_evaluation(evaluation_overview.id)
    scores = elo_score.statistics.scores
    # lower id always wins, should be sorted
    for i in range(run_count - 1):
        assert scores[run_ids[i]].elo > scores[run_ids[i + 1]].elo
        assert scores[run_ids[i]].win_rate > scores[run_ids[i + 1]].win_rate


def test_evaluate_run_only_evaluates_high_priority(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    any_instruct_output: CompleteOutput,
    argilla_fake: ArgillaFake,
) -> None:
    relevant_ids = frozenset({"1", "2"})
    evaluation_logic = InstructComparisonArgillaEvaluationLogic(relevant_ids)

    evaluator = ArgillaEvaluator(
        dataset_repository=in_memory_dataset_repository,
        run_repository=in_memory_run_repository,
        evaluation_repository=in_memory_evaluation_repository,
        description="instruct-evaluator",
        workspace_id="workspace",
        argilla_client=argilla_fake,
        evaluation_logic=evaluation_logic,
    )

    run_count = 10
    run_ids = [f"{i}" for i in range(run_count)]
    dataset_id = create_dummy_dataset(in_memory_dataset_repository)

    create_dummy_runs(
        in_memory_run_repository, any_instruct_output, run_ids, dataset_id
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
        (
            player1,
            player2,
            MatchOutcome.A_WINS,
        )
        for _ in range(10)
    ]
    elo = EloCalculator([player1, player2])
    elo.calculate(matches)

    assert elo.ratings[player1] > 1500
    assert elo.ratings[player2] < 1500

    comeback_matches = [
        (
            player1,
            player2,
            MatchOutcome.B_WINS,
        )
        for i in range(10)
    ]
    elo.calculate(comeback_matches)

    assert elo.ratings[player2] > elo.ratings[player1]
