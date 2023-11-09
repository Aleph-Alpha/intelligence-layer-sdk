from typing import Sequence

from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
)


@fixture
def qa(client: Client) -> MultipleChunkQa:
    return MultipleChunkQa(client)


CHUNK_CONTAINING_ANSWER = Chunk(
    "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. "
    "He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the "
    "forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, "
    "and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
)
RELATED_CHUNK_WITHOUT_ANSWER = Chunk(
    "In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the "
    "attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916. "
)
RELATED_QUESTION = "What is the name of Paul Nicolas' brother?"
IMPORTANT_PART_OF_CORRECT_ANSWER = "Henri"
UNRELATED_QUESTION = "What is the the capital of Germany?"


def test_multiple_chunk_qa_with_mulitple_chunks(qa: MultipleChunkQa) -> None:
    chunks: Sequence[Chunk] = [CHUNK_CONTAINING_ANSWER, RELATED_CHUNK_WITHOUT_ANSWER]

    input = MultipleChunkQaInput(chunks=chunks, question=RELATED_QUESTION)
    output = qa.run(input, NoOpTracer())

    assert output.answer
    assert IMPORTANT_PART_OF_CORRECT_ANSWER in output.answer
    assert len(output.subanswers) == 1
    assert output.subanswers[0].chunk == chunks[0]
    assert any(
        IMPORTANT_PART_OF_CORRECT_ANSWER in highlight
        for highlight in output.subanswers[0].highlights
    )


def test_multiple_chunk_qa_without_answer(qa: MultipleChunkQa) -> None:
    chunks: Sequence[Chunk] = [CHUNK_CONTAINING_ANSWER]

    input = MultipleChunkQaInput(chunks=chunks, question=UNRELATED_QUESTION)
    output = qa.run(input, NoOpTracer())

    assert output.answer is None


def test_multiple_chunk_qa_with_spanish_question(qa: MultipleChunkQa) -> None:
    question = "¿Cómo se llama el hermano de Paul Nicola?"
    chunks = [CHUNK_CONTAINING_ANSWER, CHUNK_CONTAINING_ANSWER]

    input = MultipleChunkQaInput(
        chunks=chunks, question=question, language=Language("es")
    )
    output = qa.run(input, NoOpTracer())

    assert len(output.subanswers) == len(chunks)
    assert output.answer
    assert "hermano" in output.answer
