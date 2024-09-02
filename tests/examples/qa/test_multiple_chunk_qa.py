from collections.abc import Sequence

from intelligence_layer.core import NoOpTracer
from intelligence_layer.core.chunk import TextChunk
from intelligence_layer.core.detect_language import Language
from intelligence_layer.examples.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
)

CHUNK_CONTAINING_ANSWER = TextChunk(
    "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. "
    "He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the "
    "forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, "
    "and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
)
RELATED_CHUNK_WITHOUT_ANSWER = TextChunk(
    "In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the "
    "attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916. "
)
RELATED_QUESTION = "What is the name of Paul Nicolas' brother?"
IMPORTANT_PART_OF_CORRECT_ANSWER = "Henri"
UNRELATED_QUESTION = "What is the the capital of Germany?"


def test_multiple_chunk_qa_with_mulitple_chunks(
    multiple_chunk_qa: MultipleChunkQa,
) -> None:
    chunks: Sequence[TextChunk] = [
        CHUNK_CONTAINING_ANSWER,
        RELATED_CHUNK_WITHOUT_ANSWER,
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=RELATED_QUESTION, explainability_enabled=True)
    output = multiple_chunk_qa.run(input, NoOpTracer())

    assert output.answer
    assert IMPORTANT_PART_OF_CORRECT_ANSWER in output.answer
    assert len(output.subanswers) == 1
    assert output.subanswers[0].chunk == chunks[0]
    assert any(
        IMPORTANT_PART_OF_CORRECT_ANSWER
        in CHUNK_CONTAINING_ANSWER[highlight.start : highlight.end]
        for highlight in output.subanswers[0].highlights
    )


def test_multiple_chunk_qa_without_answer(multiple_chunk_qa: MultipleChunkQa) -> None:
    chunks: Sequence[TextChunk] = [CHUNK_CONTAINING_ANSWER]

    input = MultipleChunkQaInput(chunks=chunks, question=UNRELATED_QUESTION)
    output = multiple_chunk_qa.run(input, NoOpTracer())

    assert output.answer is None


def test_multiple_chunk_qa_with_spanish_question(
    multiple_chunk_qa: MultipleChunkQa,
) -> None:
    question = "¿Cómo se llama el hermano de Paul Nicola?"
    chunks = [CHUNK_CONTAINING_ANSWER, CHUNK_CONTAINING_ANSWER]

    input = MultipleChunkQaInput(
        chunks=chunks, question=question, language=Language("es")
    )
    output = multiple_chunk_qa.run(input, NoOpTracer())

    assert len(output.subanswers) == len(chunks)
    assert output.answer
    assert "hermano" in output.answer
