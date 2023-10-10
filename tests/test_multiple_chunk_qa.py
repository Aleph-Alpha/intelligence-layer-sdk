from intelligence_layer.multiple_chunk_qa import MultipleChunkQa, MultipleChunkQaInput
from aleph_alpha_client import Client
from pytest import fixture
from pprint import pprint


@fixture
def qa(client: Client) -> MultipleChunkQa:
    return MultipleChunkQa(client, "debug")


def test_multiple_chunk_qa_with_answer(qa: MultipleChunkQa) -> None:
    question = "What is the name of Paul Nicolas' brother?"

    chunks = [
        "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=question)
    output = qa.run(input)

    assert output.answer
    assert "Henri" in output.answer


def test_multiple_chunk_qa_without_answer(qa: MultipleChunkQa) -> None:
    question = "What is the the capital of Germany?"

    chunks = [
        "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=question)
    output = qa.run(input)

    print(output.answer)
    pprint(output.debug_log)

    assert output.answer is None


def test_multiple_chunk_qa_with_mulitple_chunks(qa: MultipleChunkQa) -> None:
    question = "What is the name of Paul Nicolas' brother?"

    chunks = [
        "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3]",
        "In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916.",
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=question)
    output = qa.run(input)

    assert output.answer
    assert "Henri" in output.answer
