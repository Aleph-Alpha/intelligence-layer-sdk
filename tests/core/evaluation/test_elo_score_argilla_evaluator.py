from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Iterable, Sequence
from uuid import uuid4

from aleph_alpha_client import CompletionResponse, Prompt
from aleph_alpha_client.completion import CompletionResult
from faker import Faker
from pytest import fixture, mark

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core.complete import InstructInput, PromptOutput
from intelligence_layer.core.evaluation.dataset_repository import (
    InMemoryDatasetRepository,
)
from intelligence_layer.core.evaluation.domain import (
    Example,
    ExampleOutput,
    RunOverview,
)
from intelligence_layer.core.evaluation.elo_score_argilla_evaluator import (
    Elo,
    EloScoreArgillaEvaluator,
    Payoff,
    PayoffMatrix,
)
from intelligence_layer.core.evaluation.evaluation_repository import (
    InMemoryEvaluationRepository,
)
from intelligence_layer.core.evaluation.evaluator import ArgillaEvaluationRepository
from intelligence_layer.core.prompt_template import PromptWithMetadata
from intelligence_layer.core.tracer import utc_now


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
                    "winner": 1
                    if int(r.metadata["first"]) < int(r.metadata["second"])
                    else 2
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
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    argilla_fake: ArgillaClient,
) -> EloScoreArgillaEvaluator:
    eval_repository = ArgillaEvaluationRepository(
        in_memory_evaluation_repository, argilla_fake
    )
    return EloScoreArgillaEvaluator(
        eval_repository, in_memory_dataset_repository, "workspace"
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
        prompt_with_metadata=PromptWithMetadata(prompt=Prompt([]), ranges={}),
    )


def test_evaluate_run_submits_pairwise_comparison_records(
    evaluator: EloScoreArgillaEvaluator,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    any_instruct_output: PromptOutput,
    argilla_fake: ArgillaFake,
) -> None:
    instruct_completion = any_instruct_output.completion
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
        in_memory_evaluation_repository.store_example_output(
            example_output=ExampleOutput(
                run_id=run_id, example_id="example_id", output=any_instruct_output
            )
        )
        in_memory_evaluation_repository.store_run_overview(
            RunOverview(
                dataset_id=dataset_id,
                id=run_id,
                start=utc_now(),
                end=utc_now(),
                failed_example_count=0,
                successful_example_count=0,
                runner_id="runner",
            )
        )

    evaluation_overview = evaluator.evaluate_runs(*run_ids)

    pairs = combinations(run_ids, 2)
    assert argilla_fake.record_data(evaluation_overview.id) == [
        RecordData(
            content={
                "instruction": instruction,
                "input": instruction_input,
                "first": instruct_completion,
                "second": instruct_completion,
            },
            example_id=example_id,
            metadata={"first": first, "second": second},
        )
        for [first, second] in pairs
    ]

    elo_score = evaluator.aggregate_evaluation(evaluation_overview.id)
    scores = elo_score.statistics.elos
    # lower id always wins, should be sorted
    for i in range(run_count - 1):
        assert scores[run_ids[i]] > scores[run_ids[i + 1]]


@mark.skip
def test_evaluate_run_only_evaluates_high_priority(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    any_instruct_output: PromptOutput,
    argilla_fake: ArgillaFake,
) -> None:
    # TODO
    # To the people potentially debugging this:
    # in_memory_evaluation_repository_1, despite being instatiated below, will have an evaluation_run_overview
    # no idea why
    in_memory_evaluation_repository_1 = InMemoryEvaluationRepository()
    in_memory_dataset_repository_1 = InMemoryDatasetRepository()
    argilla_fake_1 = ArgillaFake()
    print("test_evaluate_run_only_evaluates_high_priority")
    print(f"{in_memory_dataset_repository_1=}, {in_memory_evaluation_repository_1=}")
    print(
        f"{id(in_memory_dataset_repository_1)=}, {id(in_memory_evaluation_repository_1)=}"
    )
    eval_repository = ArgillaEvaluationRepository(
        in_memory_evaluation_repository_1, argilla_fake_1
    )
    relevant_ids = frozenset({"1", "2"})
    evaluator = EloScoreArgillaEvaluator(
        eval_repository, in_memory_dataset_repository_1, "workspace", relevant_ids
    )

    run_count = 10
    run_ids = [f"{i}" for i in range(run_count)]
    example_id = "example_id"
    instruction = "inst"
    instruction_input = "some text"
    dataset_id = in_memory_dataset_repository_1.create_dataset(
        [
            Example(
                id=example_id,
                input=InstructInput(instruction=instruction, input=instruction_input),
                expected_output=None,
            )
        ]
    )
    for run_id in run_ids:
        in_memory_evaluation_repository_1.store_example_output(
            example_output=ExampleOutput(
                run_id=run_id, example_id="example_id", output=any_instruct_output
            )
        )
        in_memory_evaluation_repository_1.store_run_overview(
            RunOverview(
                dataset_id=dataset_id,
                id=run_id,
                start=utc_now(),
                end=utc_now(),
                failed_example_count=0,
                successful_example_count=0,
                runner_id="runner",
            )
        )

    evaluation_overview = evaluator.evaluate_runs(*run_ids)

    def relevant_ids_in_record(record: RecordData) -> bool:
        players = [record.metadata["first"], record.metadata["second"]]
        return any(id in players for id in relevant_ids)

    records = argilla_fake_1.record_data(evaluation_overview.id)
    assert all(relevant_ids_in_record(record) for record in records)
    assert len(records) == sum(run_count - (i + 1) for i in range(len(relevant_ids)))


# This test only exists because the above is failing for mysterious reasons
def test_evaluate_run_only_evaluates_high_priority_2(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_dataset_repository: InMemoryDatasetRepository,
    any_instruct_output: PromptOutput,
    argilla_fake: ArgillaFake,
) -> None:
    eval_repository = ArgillaEvaluationRepository(
        in_memory_evaluation_repository, argilla_fake
    )

    run_count = 10
    run_ids = [str(uuid4()) for i in range(run_count)]
    relevant_ids = frozenset({run_ids[0], run_ids[1]})
    evaluator = EloScoreArgillaEvaluator(
        eval_repository, in_memory_dataset_repository, "workspace", relevant_ids
    )

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
        in_memory_evaluation_repository.store_example_output(
            example_output=ExampleOutput(
                run_id=run_id, example_id="example_id", output=any_instruct_output
            )
        )
        in_memory_evaluation_repository.store_run_overview(
            RunOverview(
                dataset_id=dataset_id,
                id=run_id,
                start=utc_now(),
                end=utc_now(),
                failed_example_count=0,
                successful_example_count=0,
                runner_id="runner",
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
    elo = Elo([player1, player2])
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
