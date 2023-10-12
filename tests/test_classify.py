from typing import Sequence
from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.classify import (
    SingleLabelClassify,
    ClassifyInput,
    ClassifyOutput,
    SingleLabelClassifyEvaluator,
)
from intelligence_layer.task import (
    NoOpDebugLogger,
)


@fixture
def single_label_classify(client: Client) -> SingleLabelClassify:
    return SingleLabelClassify(client)


def test_single_label_classify_returns_score_for_all_labels(
    single_label_classify: SingleLabelClassify,
) -> None:
    classify_input = ClassifyInput(
        text="This is good",
        labels=frozenset({"positive", "negative"}),
    )

    classify_output = single_label_classify.run(classify_input, NoOpDebugLogger())

    # Output contains everything we expect
    assert isinstance(classify_output, ClassifyOutput)
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_single_label_classify_accomodates_labels_starting_with_spaces(
    single_label_classify: SingleLabelClassify,
) -> None:
    classify_input = ClassifyInput(
        text="This is good", labels=frozenset({" positive", "negative"})
    )

    classify_output = single_label_classify.run(classify_input, NoOpDebugLogger())

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_single_label_classify_accomodates_labels_starting_with_different_spaces(
    single_label_classify: SingleLabelClassify,
) -> None:
    classify_input = ClassifyInput(
        text="This is good", labels=frozenset({" positive", "  positive"})
    )

    classify_output = single_label_classify.run(classify_input, NoOpDebugLogger())

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)
    assert classify_output.scores[" positive"] != classify_output.scores["  positive"]


def test_single_label_classify_sentiment_classification(
    single_label_classify: SingleLabelClassify,
) -> None:
    classify_input = ClassifyInput(
        text="This is good", labels=frozenset({"positive", "negative"})
    )

    classify_output = single_label_classify.run(classify_input, NoOpDebugLogger())

    # Verify we got a higher positive score
    assert classify_output.scores["positive"] > classify_output.scores["negative"]


def test_single_label_classify_emotion_classification(
    single_label_classify: SingleLabelClassify,
) -> None:
    classify_input = ClassifyInput(
        text="I love my job", labels=frozenset({"happy", "sad", "frustrated", "angry"})
    )

    classify_output = single_label_classify.run(classify_input, NoOpDebugLogger())

    # Verify it correctly calculated happy
    assert classify_output.scores["happy"] == max(classify_output.scores.values())


def test_single_label_classify_handles_labels_starting_with_same_token(
    single_label_classify: SingleLabelClassify,
) -> None:
    classify_input = ClassifyInput(
        text="This is good",
        labels=frozenset({"positive", "positive positive"}),
    )

    classify_output = single_label_classify.run(classify_input, NoOpDebugLogger())

    assert classify_input.labels == set(r for r in classify_output.scores)


def test_can_evaluate_classify(single_label_classify: SingleLabelClassify) -> None:
    classify_input = ClassifyInput(
        text="This is good",
        labels=frozenset({"positive", "negative"}),
    )
    evaluator = SingleLabelClassifyEvaluator(task=single_label_classify)

    evaluation = evaluator.evaluate(
        input=classify_input, logger=NoOpDebugLogger(), expected_output=["positive"]
    )

    assert evaluation.correct == True


def test_can_aggregate_evaluations(
    single_label_classify: SingleLabelClassify,
) -> None:
    # Mypy doesn't recognize a sequence of tuples, so we ignore the error here
    correct_input: tuple[ClassifyInput, Sequence[str]] = [
        ClassifyInput(
            text="This is good",
            labels=frozenset({"positive", "negative"}),
        ),
        ["positive"],
    ]  # type: ignore
    incorrect_input: tuple[ClassifyInput, Sequence[str]] = [
        ClassifyInput(
            text="This is extremely bad",
            labels=frozenset({"positive", "negative"}),
        ),
        ["positive"],
    ]  # type: ignore

    single_label_classify_evaluator = SingleLabelClassifyEvaluator(
        task=single_label_classify
    )

    aggregated_evaluations = single_label_classify_evaluator.evaluate_dataset(
        [correct_input, incorrect_input], logger=NoOpDebugLogger()
    )

    assert aggregated_evaluations.percentage_correct == 0.5


def test_aggregating_evaluations_works_with_empty_list(
    single_label_classify: SingleLabelClassify,
) -> None:
    single_label_classify_evaluator = SingleLabelClassifyEvaluator(
        task=single_label_classify
    )

    aggregated_evaluations = single_label_classify_evaluator.evaluate_dataset(
        [], logger=NoOpDebugLogger()
    )

    assert aggregated_evaluations.percentage_correct == 0
