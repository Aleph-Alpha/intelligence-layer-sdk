from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.core.task import Chunk
from intelligence_layer.use_cases.classify.classify import (
    ClassifyInput,
    ClassifyOutput,
)
from intelligence_layer.use_cases.classify.embedding_based_classify import (
    LabelWithExamples,
    EmbeddingBasedClassify,
)


@fixture
def embedding_based_classify(client: Client) -> EmbeddingBasedClassify:
    classes_with_examples = [
        LabelWithExamples(
            name="positive",
            examples=[
                "I really like this.",
                "Wow, your hair looks great!",
                "We're so in love."
            ]
        ),
        LabelWithExamples(
            name="negative",
            examples=[
                "I really dislike this.",
                "Ugh, Your hair looks horrible!",
                "We're not in love anymore."
            ]
        )
    ]
    return EmbeddingBasedClassify(classes_with_examples, client)


def test_embedding_based_classify_returns_score_for_all_labels(
    embedding_based_classify: EmbeddingBasedClassify,
) -> None:
    classify_input = ClassifyInput(
        chunk=Chunk("This is good"),
        labels=frozenset({"positive", "negative"}),
    )

    classify_output = embedding_based_classify.run(classify_input, NoOpDebugLogger())

    # Output contains everything we expect
    assert isinstance(classify_output, ClassifyOutput)
    assert classify_input.labels == set(r for r in classify_output.scores)
