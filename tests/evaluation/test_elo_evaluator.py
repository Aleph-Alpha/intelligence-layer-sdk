from typing import Sequence, Tuple

from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import (
    ControlModel,
    Language,
    LuminousControlModel,
    TextChunk,
    utc_now,
)
from intelligence_layer.core.tracer.tracer import NoOpTracer, Tracer
from intelligence_layer.evaluation import (
    ComparisonEvaluation,
    EloEvaluationLogic,
    EvaluationLogic,
    Evaluator,
    Example,
    ExampleOutput,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    InMemoryRunRepository,
    Matches,
    MatchOutcome,
    RunOverview,
    SuccessfulExampleOutput,
)
from intelligence_layer.examples import SingleChunkQaInput, SingleChunkQaOutput

load_dotenv()


class DummyEloQaEvalLogic(
    EloEvaluationLogic[SingleChunkQaInput, SingleChunkQaOutput, SingleChunkQaOutput]
):
    def __init__(
        self,
        model: ControlModel,
        tracer: Tracer = NoOpTracer(),
    ):
        super().__init__()
        self._model = model
        self.tracer = tracer

    def grade(
        self,
        first: SuccessfulExampleOutput[SingleChunkQaOutput],
        second: SuccessfulExampleOutput[SingleChunkQaOutput],
        example: Example[SingleChunkQaInput, SingleChunkQaOutput],
    ) -> MatchOutcome:
        _ = example
        if first.run_id < second.run_id:
            return MatchOutcome.A_WINS
        elif first.run_id > second.run_id:
            return MatchOutcome.B_WINS
        else:
            return MatchOutcome.DRAW


@fixture
def model(client: AlephAlphaClientProtocol) -> ControlModel:
    return LuminousControlModel(client=client, name="luminous-base-control")


@fixture
def in_memory_dataset_repository() -> InMemoryDatasetRepository:
    return InMemoryDatasetRepository()


@fixture
def in_memory_run_repository() -> InMemoryRunRepository:
    return InMemoryRunRepository()


@fixture
def in_memory_evaluation_repository() -> InMemoryEvaluationRepository:
    return InMemoryEvaluationRepository()


@fixture
def dummy_eval_logic(model: ControlModel) -> DummyEloQaEvalLogic:
    return DummyEloQaEvalLogic(model=model)


@fixture
def elo_evaluator(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    dummy_eval_logic: EvaluationLogic[
        SingleChunkQaInput, SingleChunkQaOutput, SingleChunkQaOutput, Matches
    ],
) -> Evaluator[SingleChunkQaInput, SingleChunkQaOutput, SingleChunkQaOutput, Matches]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "Testing",
        dummy_eval_logic,
    )


@fixture
def dummy_qa_input() -> SingleChunkQaInput:
    return SingleChunkQaInput(chunk=TextChunk(""), question="", language=Language("en"))


@fixture
def dummy_qa_output() -> SingleChunkQaOutput:
    return SingleChunkQaOutput(answer=None, highlights=[])


@fixture
def qa_outputs() -> Sequence[SingleChunkQaOutput]:
    return [
        SingleChunkQaOutput(answer=answer, highlights=[])
        for answer in [
            "Surface micromachining builds microstructures.",
            "Surface micromachining builds microstructures. This is done by deposition and etching structural layers over a substrate.",
            "Surface micromachining builds microstructures by deposition and etching structural layers over a substrate. This is different from Bulk micromachining, in which a silicon substrate wafer is selectively etched to produce structures.",
        ]
    ]


@fixture
def qa_setup(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    in_memory_run_repository: InMemoryRunRepository,
    qa_outputs: Sequence[SingleChunkQaOutput],
) -> Tuple[Sequence[str], str]:
    qa_input_text = TextChunk(
        """Surface micromachining builds microstructures by deposition and etching structural layers over a substrate.[1] This is different from Bulk micromachining, in which a silicon substrate wafer is selectively etched to produce structures."""
    )
    #
    qa_input = SingleChunkQaInput(
        chunk=qa_input_text, question="What is micromachining?", language=Language("en")
    )
    expected_output = "Surface micromachining builds microstructures by deposition and etching structural layers over a substrate."
    #
    example_id = "some-example-id"
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[
            Example(input=qa_input, expected_output=expected_output, id=example_id)
        ],
        dataset_name="some-example-dataset-name",
    ).id
    #
    run_ids = [f"some-run-id-{i}" for i in range(len(qa_outputs))]
    for i, output in enumerate(qa_outputs):
        in_memory_run_repository.store_example_output(
            example_output=ExampleOutput(
                run_id=run_ids[i],
                example_id=example_id,
                output=output,
            )
        )
        in_memory_run_repository.store_run_overview(
            RunOverview(
                dataset_id=dataset_id,
                id=run_ids[i],
                start=utc_now(),
                end=utc_now(),
                failed_example_count=0,
                successful_example_count=len(qa_outputs),
                description="runner",
            )
        )
    return run_ids, dataset_id


def test_evaluate_runs_creates_correct_matches_for_elo_qa_eval(
    qa_setup: Tuple[Sequence[str], str],
    elo_evaluator: Evaluator[
        SingleChunkQaInput, SingleChunkQaOutput, SingleChunkQaOutput, Matches
    ],
) -> None:
    run_ids, _ = qa_setup
    evaluation_overview = elo_evaluator.evaluate_runs(*run_ids)

    eval_result = list(elo_evaluator.evaluation_lineages(evaluation_overview.id))[
        0
    ].evaluation.result
    assert isinstance(eval_result, Matches)
    matches = eval_result.comparison_evaluations

    for match in matches:
        assert isinstance(match, ComparisonEvaluation)
        if match.first_player < match.second_player:
            assert match.outcome == MatchOutcome.A_WINS
        elif match.first_player > match.second_player:
            assert match.outcome == MatchOutcome.B_WINS
