from typing import Sequence

from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core import (
    Example,
    InMemoryEvaluationRepository,
    SequenceDataset,
)
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.tracer import InMemoryTracer, NoOpTracer
from intelligence_layer.use_cases.classify.classify import (
    ClassifyInput,
    SingleLabelClassifyEvaluation,
    SingleLabelClassifyEvaluator,
    SingleLabelClassifyOutput,
)
from intelligence_layer.use_cases.classify.prompt_based_classify import (
    PromptBasedClassify,
)


@fixture
def prompt_based_classify(client: AlephAlphaClientProtocol) -> PromptBasedClassify:
    return PromptBasedClassify(client)


@fixture
def classify_evaluator(
    prompt_based_classify: PromptBasedClassify,
) -> SingleLabelClassifyEvaluator:
    return SingleLabelClassifyEvaluator(
        prompt_based_classify, InMemoryEvaluationRepository()
    )


def test_prompt_based_classify_returns_score_for_all_labels(
    prompt_based_classify: PromptBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Output contains everything we expect
    assert isinstance(classify_output, SingleLabelClassifyOutput)
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_prompt_based_classify_accomodates_labels_starting_with_spaces(
    prompt_based_classify: PromptBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"), labels=frozenset({" positive", "negative"})
    )

    tracer = InMemoryTracer()
    classify_output = prompt_based_classify.run(classify_input, tracer)

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_prompt_based_classify_accomodates_labels_starting_with_different_spaces(
    prompt_based_classify: PromptBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"), labels=frozenset({" positive", "  positive"})
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)
    assert classify_output.scores[" positive"] != classify_output.scores["  positive"]


def test_prompt_based_classify_sentiment_classification(
    prompt_based_classify: PromptBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"), labels=frozenset({"positive", "negative"})
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Verify we got a higher positive score
    assert classify_output.scores["positive"] > classify_output.scores["negative"]


def test_prompt_based_classify_emotion_classification(
    prompt_based_classify: PromptBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("I love my job"),
        labels=frozenset({"happy", "sad", "frustrated", "angry"}),
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    # Verify it correctly calculated happy
    assert classify_output.scores["happy"] == max(classify_output.scores.values())


def test_prompt_based_classify_handles_labels_starting_with_same_token(
    prompt_based_classify: PromptBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "positive positive"}),
    )

    classify_output = prompt_based_classify.run(classify_input, NoOpTracer())

    assert classify_input.labels == set(r for r in classify_output.scores)


def test_can_evaluate_classify(
    classify_evaluator: SingleLabelClassifyEvaluator,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )

    evaluation = classify_evaluator.evaluate(
        input=classify_input,
        tracer=NoOpTracer(),
        expected_output=["positive"],
    )

    assert isinstance(evaluation, SingleLabelClassifyEvaluation)
    assert evaluation.correct is True


def test_can_aggregate_evaluations(
    classify_evaluator: SingleLabelClassifyEvaluator,
) -> None:
    positive_lst: Sequence[str] = ["positive"]
    correct_example = Example(
        input=ClassifyInput(
            chunk=Chunk("This is good"),
            labels=frozenset({"positive", "negative"}),
        ),
        expected_output=positive_lst,
    )
    incorrect_example = Example(
        input=ClassifyInput(
            chunk=Chunk("This is extremely bad"),
            labels=frozenset({"positive", "negative"}),
        ),
        expected_output=positive_lst,
    )

    dataset = SequenceDataset(
        name="classify_test", examples=[correct_example, incorrect_example]
    )

    evaluation_overview = classify_evaluator.evaluate_dataset(
        dataset, tracer=NoOpTracer()
    )

    assert evaluation_overview.statistics.percentage_correct == 0.5


def test_aggregating_evaluations_works_with_empty_list(
    classify_evaluator: SingleLabelClassifyEvaluator,
) -> None:
    evaluation_overview = classify_evaluator.evaluate_dataset(
        SequenceDataset(name="empty_dataset", examples=[]), tracer=NoOpTracer()
    )

    assert evaluation_overview.statistics.percentage_correct == 0
