from collections.abc import Sequence

import pytest
from _pytest.fixtures import FixtureRequest
from pytest import fixture

from intelligence_layer.core import InMemoryTracer, NoOpTracer, TextChunk
from intelligence_layer.core.model import Llama3InstructModel, LuminousControlModel
from intelligence_layer.evaluation import (
    Aggregator,
    DatasetRepository,
    Evaluator,
    Example,
    InMemoryAggregationRepository,
    InMemoryDatasetRepository,
    InMemoryEvaluationRepository,
    Runner,
    RunRepository,
)
from intelligence_layer.examples.classify.classify import (
    AggregatedSingleLabelClassifyEvaluation,
    ClassifyInput,
    SingleLabelClassifyAggregationLogic,
    SingleLabelClassifyEvaluation,
    SingleLabelClassifyEvaluationLogic,
    SingleLabelClassifyOutput,
)
from intelligence_layer.examples.classify.prompt_based_classify import (
    PromptBasedClassify,
)


@fixture
def prompt_based_classify_luminous(
    luminous_control_model: LuminousControlModel,
) -> PromptBasedClassify:
    return PromptBasedClassify(luminous_control_model)


@fixture
def prompt_based_classify_llama3(
    luminous_control_model: Llama3InstructModel,
) -> PromptBasedClassify:
    return PromptBasedClassify(Llama3InstructModel())


@fixture
def single_label_classify_eval_logic() -> SingleLabelClassifyEvaluationLogic:
    return SingleLabelClassifyEvaluationLogic()


@fixture
def single_label_classify_aggregation_logic() -> SingleLabelClassifyAggregationLogic:
    return SingleLabelClassifyAggregationLogic()


@fixture
def classify_evaluator(
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: RunRepository,
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    single_label_classify_eval_logic: SingleLabelClassifyEvaluationLogic,
) -> Evaluator[
    ClassifyInput,
    SingleLabelClassifyOutput,
    str,
    SingleLabelClassifyEvaluation,
]:
    return Evaluator(
        in_memory_dataset_repository,
        in_memory_run_repository,
        in_memory_evaluation_repository,
        "single-label-classify",
        single_label_classify_eval_logic,
    )


@fixture
def classify_aggregator(
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    in_memory_aggregation_repository: InMemoryAggregationRepository,
    single_label_classify_aggregation_logic: SingleLabelClassifyAggregationLogic,
) -> Aggregator[
    SingleLabelClassifyEvaluation,
    AggregatedSingleLabelClassifyEvaluation,
]:
    return Aggregator(
        in_memory_evaluation_repository,
        in_memory_aggregation_repository,
        "single-label-classify",
        single_label_classify_aggregation_logic,
    )


@fixture
def classify_runner(
    prompt_based_classify_luminous: PromptBasedClassify,
    in_memory_dataset_repository: DatasetRepository,
    in_memory_run_repository: RunRepository,
) -> Runner[ClassifyInput, SingleLabelClassifyOutput]:
    return Runner(
        prompt_based_classify_luminous,
        in_memory_dataset_repository,
        in_memory_run_repository,
        "prompt-based-classify",
    )


@pytest.mark.parametrize(
    "prompt_based_classify_model",
    ["prompt_based_classify_luminous", "prompt_based_classify_llama3"],
)
def test_prompt_based_classify_returns_score_for_all_labels(
    prompt_based_classify_model: str, request: FixtureRequest
) -> None:
    prompt_based_classify = request.getfixturevalue(prompt_based_classify_model)
    classify_input = ClassifyInput(
        chunk=TextChunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Output contains everything we expect
    assert isinstance(classify_output, SingleLabelClassifyOutput)
    assert classify_input.labels == set(r for r in classify_output.scores)


@pytest.mark.parametrize(
    "prompt_based_classify_model",
    ["prompt_based_classify_luminous", "prompt_based_classify_llama3"],
)
def test_prompt_based_classify_accomodates_labels_starting_with_spaces(
    prompt_based_classify_model: str,
    request: FixtureRequest,
) -> None:
    prompt_based_classify = request.getfixturevalue(prompt_based_classify_model)
    classify_input = ClassifyInput(
        chunk=TextChunk("This is good"), labels=frozenset({" positive", "negative"})
    )

    tracer = InMemoryTracer()
    classify_output = prompt_based_classify.run(classify_input, tracer)

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)


@pytest.mark.parametrize(
    "prompt_based_classify_model",
    ["prompt_based_classify_luminous", "prompt_based_classify_llama3"],
)
def test_prompt_based_classify_accomodates_labels_starting_with_different_spaces(
    prompt_based_classify_model: str,
    request: FixtureRequest,
) -> None:
    prompt_based_classify = request.getfixturevalue(prompt_based_classify_model)
    classify_input = ClassifyInput(
        chunk=TextChunk("This is good"), labels=frozenset({" positive", "  positive"})
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)
    assert classify_output.scores[" positive"] != classify_output.scores["  positive"]


@pytest.mark.parametrize(
    "prompt_based_classify_model",
    ["prompt_based_classify_luminous", "prompt_based_classify_llama3"],
)
def test_prompt_based_classify_sentiment_classification(
    prompt_based_classify_model: str, request: FixtureRequest
) -> None:
    prompt_based_classify = request.getfixturevalue(prompt_based_classify_model)
    classify_input = ClassifyInput(
        chunk=TextChunk("This is good"), labels=frozenset({"positive", "negative"})
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Verify we got a higher positive score
    assert classify_output.scores["positive"] > classify_output.scores["negative"]


@pytest.mark.parametrize(
    "prompt_based_classify_model",
    ["prompt_based_classify_luminous", "prompt_based_classify_llama3"],
)
def test_prompt_based_classify_emotion_classification(
    prompt_based_classify_model: str,
    request: FixtureRequest,
) -> None:
    prompt_based_classify = request.getfixturevalue(prompt_based_classify_model)
    classify_input = ClassifyInput(
        chunk=TextChunk("I love my job"),
        labels=frozenset({"happy", "sad", "frustrated", "angry"}),
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Verify it correctly calculated happy
    assert classify_output.scores["happy"] == max(classify_output.scores.values())


@pytest.mark.parametrize(
    "prompt_based_classify_model",
    ["prompt_based_classify_luminous", "prompt_based_classify_llama3"],
)
def test_prompt_based_classify_handles_labels_starting_with_same_token(
    prompt_based_classify_model: str, request: FixtureRequest
) -> None:
    prompt_based_classify = request.getfixturevalue(prompt_based_classify_model)
    classify_input = ClassifyInput(
        chunk=TextChunk("This is good"),
        labels=frozenset({"positive", "positive positive"}),
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    assert classify_input.labels == set(r for r in classify_output.scores)


def test_can_evaluate_classify(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    classify_runner: Runner[ClassifyInput, SingleLabelClassifyOutput],
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    classify_evaluator: Evaluator[
        ClassifyInput,
        SingleLabelClassifyOutput,
        Sequence[str],
        SingleLabelClassifyEvaluation,
    ],
    prompt_based_classify_luminous: PromptBasedClassify,
) -> None:
    example = Example(
        input=ClassifyInput(
            chunk=TextChunk("This is good"),
            labels=frozenset({"positive", "negative"}),
        ),
        expected_output="positive",
    )

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[example], dataset_name="test-dataset"
    ).id

    run_overview = classify_runner.run_dataset(dataset_id)
    evaluation_overview = classify_evaluator.evaluate_runs(run_overview.id)

    evaluation = in_memory_evaluation_repository.example_evaluations(
        evaluation_overview.id,
        SingleLabelClassifyEvaluation,
    )[0].result

    assert isinstance(evaluation, SingleLabelClassifyEvaluation)
    assert evaluation.correct is True


def test_classify_warns_on_missing_label(
    in_memory_dataset_repository: InMemoryDatasetRepository,
    classify_runner: Runner[ClassifyInput, SingleLabelClassifyOutput],
    in_memory_evaluation_repository: InMemoryEvaluationRepository,
    classify_evaluator: Evaluator[
        ClassifyInput,
        SingleLabelClassifyOutput,
        Sequence[str],
        SingleLabelClassifyEvaluation,
    ],
    prompt_based_classify_luminous: PromptBasedClassify,
) -> None:
    example = Example(
        input=ClassifyInput(
            chunk=TextChunk("This is good"),
            labels=frozenset({"positive", "negative"}),
        ),
        expected_output="SomethingElse",
    )

    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[example], dataset_name="test-dataset"
    ).id

    run_overview = classify_runner.run_dataset(dataset_id)

    pytest.warns(RuntimeWarning, classify_evaluator.evaluate_runs, run_overview.id)


def test_can_aggregate_evaluations(
    classify_evaluator: Evaluator[
        ClassifyInput,
        SingleLabelClassifyOutput,
        Sequence[str],
        SingleLabelClassifyEvaluation,
    ],
    classify_aggregator: Aggregator[
        SingleLabelClassifyEvaluation,
        AggregatedSingleLabelClassifyEvaluation,
    ],
    in_memory_dataset_repository: InMemoryDatasetRepository,
    classify_runner: Runner[ClassifyInput, SingleLabelClassifyOutput],
) -> None:
    positive: str = "positive"
    correct_example = Example(
        input=ClassifyInput(
            chunk=TextChunk("This is good"),
            labels=frozenset({"positive", "negative"}),
        ),
        expected_output=positive,
    )
    incorrect_example = Example(
        input=ClassifyInput(
            chunk=TextChunk("This is extremely bad"),
            labels=frozenset({"positive", "negative"}),
        ),
        expected_output=positive,
    )
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[correct_example, incorrect_example], dataset_name="test-dataset"
    ).id

    run_overview = classify_runner.run_dataset(dataset_id)
    evaluation_overview = classify_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = classify_aggregator.aggregate_evaluation(
        evaluation_overview.id
    )

    assert aggregation_overview.statistics.percentage_correct == 0.5
    assert aggregation_overview.statistics.confusion_matrix == {
        "positive": {"positive": 1, "negative": 0},
        "negative": {"positive": 1, "negative": 0},
    }
    assert aggregation_overview.statistics.recall_by_class["positive"] == 1 / 2
    assert aggregation_overview.statistics.precision_by_class["positive"] == 1
    assert aggregation_overview.statistics.f1_by_class["positive"] == 2 / 3
    assert aggregation_overview.statistics.precision_by_class["negative"] == 0
    assert aggregation_overview.statistics.recall_by_class["negative"] is None
    assert aggregation_overview.statistics.f1_by_class["negative"] == 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_aggregating_evaluations_works_with_empty_list(
    classify_evaluator: Evaluator[
        ClassifyInput,
        SingleLabelClassifyOutput,
        Sequence[str],
        SingleLabelClassifyEvaluation,
    ],
    classify_aggregator: Aggregator[
        SingleLabelClassifyEvaluation,
        AggregatedSingleLabelClassifyEvaluation,
    ],
    classify_runner: Runner[ClassifyInput, SingleLabelClassifyOutput],
    in_memory_dataset_repository: DatasetRepository,
) -> None:
    dataset_id = in_memory_dataset_repository.create_dataset(
        examples=[], dataset_name="test-dataset"
    ).id
    run_overview = classify_runner.run_dataset(dataset_id)
    evaluation_overview = classify_evaluator.evaluate_runs(run_overview.id)
    aggregation_overview = classify_aggregator.aggregate_evaluation(
        evaluation_overview.id
    )

    assert aggregation_overview.statistics.percentage_correct == 0
