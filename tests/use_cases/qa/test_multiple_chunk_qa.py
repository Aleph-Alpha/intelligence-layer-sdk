from typing import Sequence

from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.tracer import InMemoryTracer, NoOpTracer
from intelligence_layer.use_cases.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
)


@fixture
def qa(client: Client) -> MultipleChunkQa:
    return MultipleChunkQa(client)


def test_multiple_chunk_qa_with_answer(qa: MultipleChunkQa) -> None:
    question = "What is the name of Paul Nicolas' brother?"

    chunks: Sequence[Chunk] = [
        Chunk(
            "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
        )
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=question)
    output = qa.run(input, NoOpTracer())

    assert output.answer
    assert "Henri" in output.answer
    assert any(
        any("Henri" in highlight for highlight in source.highlights)
        for source in output.subanswers
    )
    assert len(output.subanswers) == 1


def test_multiple_chunk_qa_without_answer(qa: MultipleChunkQa) -> None:
    question = "What is the the capital of Germany?"

    chunks: Sequence[Chunk] = [
        Chunk(
            "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
        )
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=question)
    debug_log = InMemoryTracer()
    output = qa.run(input, debug_log)

    assert output.answer is None


def test_multiple_chunk_qa_with_mulitple_chunks(qa: MultipleChunkQa) -> None:
    question = "What is the name of Paul Nicolas' brother?"

    chunks: Sequence[Chunk] = [
        Chunk(
            "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3]"
        ),
        Chunk(
            "In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916. "
        ),
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=question)
    logger = InMemoryTracer()
    output = qa.run(input, logger)

    assert output.answer
    assert "Henri" in output.answer
    assert len(output.subanswers) == 1
    assert output.subanswers[0].chunk == chunks[0]
    assert any("Henri" in highlight for highlight in output.subanswers[0].highlights)
