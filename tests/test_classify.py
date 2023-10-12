import random
from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.classify import (
    Probability,
    SingleLabelClassify,
    ClassifyInput,
    ClassifyOutput,
    SingleLabelClassifyEvaluator,
)
from intelligence_layer.task import (
    DebugLogger,
    NoOpDebugLogger,
    Task,
    log_run_input_output,
)


class RandomLabelClassify(Task[ClassifyInput, ClassifyOutput]):
    @log_run_input_output
    def run(self, input: ClassifyInput, logger: DebugLogger) -> ClassifyOutput:
        return ClassifyOutput(
            scores={label: Probability(random.random()) for label in input.labels},
        )


@fixture
def random_label_classify() -> RandomLabelClassify:
    return RandomLabelClassify()


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


def test_can_compare_classifiers(
    random_label_classify: RandomLabelClassify,
    single_label_classify: SingleLabelClassify,
) -> None:
    inputs = [
        ClassifyInput(
            text="This is good",
            labels=frozenset({"positive", "negative"}),
        )
        for _ in range(3)
    ]
    single_label_classify_evaluator = SingleLabelClassifyEvaluator(
        task=single_label_classify
    )
