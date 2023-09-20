import os
from typing import Iterable

from aleph_alpha_client import Client
from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.classify import (
    SingleLabelClassify,
    ClassifyInput,
    ClassifyOutput,
)
from intelligence_layer._task import DebugLog


@fixture
def client() -> Iterable[Client]:
    """Provide fixture for api."""
    try:
        load_dotenv()
        token = os.getenv("AA_API_TOKEN")
        assert isinstance(token, str)
        yield Client(token=token)
    finally:
        pass


def test_single_label_classify_returns_score_for_all_labels(client: Client) -> None:
    classify = SingleLabelClassify(client=client)
    classify_input = ClassifyInput(text="This is good", labels={"positive", "negative"})

    classify_output = classify.run(classify_input)

    # Output contains everything we expect
    assert isinstance(classify_output, ClassifyOutput)
    assert isinstance(classify_output.debug_log, DebugLog)
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_single_label_classify_accomodates_labels_starting_with_spaces(
    client: Client,
) -> None:
    classify = SingleLabelClassify(client=client)
    classify_input = ClassifyInput(
        text="This is good", labels={" positive", "negative"}
    )

    classify_output = classify.run(classify_input)

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)


def test_single_label_classify_accomodates_labels_starting_with_different_spaces(
    client: Client,
) -> None:
    classify = SingleLabelClassify(client=client)
    classify_input = ClassifyInput(
        text="This is good", labels={" positive", "  positive"}
    )

    classify_output = classify.run(classify_input)

    # Output contains everything we expect
    assert classify_input.labels == set(r for r in classify_output.scores)
    assert classify_output.scores[" positive"] != classify_output.scores["  positive"]


def test_single_label_classify_sentiment_classification(client: Client) -> None:
    classify = SingleLabelClassify(client=client)
    classify_input = ClassifyInput(text="This is good", labels={"positive", "negative"})

    classify_output = classify.run(classify_input)

    # Verify we got a higher positive score
    assert classify_output.scores["positive"] > classify_output.scores["negative"]


def test_single_label_classify_emotion_classification(client: Client) -> None:
    classify = SingleLabelClassify(client=client)
    classify_input = ClassifyInput(
        text="I love my job", labels={"happy", "sad", "frustrated", "angry"}
    )

    classify_output = classify.run(classify_input)

    # Verify it correctly calculated happy
    assert classify_output.scores["happy"] == max(classify_output.scores.values())
