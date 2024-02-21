from pytest import fixture

from intelligence_layer.core import (
    ChunkInput,
    ChunkOverlapTask,
    InMemoryTracer,
    LuminousControlModel,
)


@fixture
def some_large_text() -> str:
    return """
  The Williamsburgh Savings Bank Tower, also known as One Hanson Place, is a skyscraper in the Fort Greene neighborhood of Brooklyn in New York City. Located at the northeast corner of Ashland Place and Hanson Place near Downtown Brooklyn, the tower was designed by Halsey, McCormack & Helmer and constructed from 1927 to 1929 as the new headquarters for the Williamsburgh Savings Bank. At 41 stories and 512 feet (156 m) tall, the Williamsburgh Savings Bank Tower was the tallest building in Brooklyn until 2009.

  The Williamsburgh Savings Bank was originally headquartered in Williamsburg, Brooklyn; its officers decided to construct a new skyscraper headquarters near Downtown Brooklyn in the mid-1920s. The bank occupied the lowest floors when the building opened on April 1, 1929, while the remaining stories were rented as offices. By the late 20th century, dentists' offices occupied much of the structure. The New York City Landmarks Preservation Commission designated the tower's exterior as a city landmark in 1977 and designated some of the interior spaces in 1996. Through several mergers, the Williamsburgh Savings Bank became part of HSBC Bank USA, which sold the building in 2004. The building's upper stories were converted to luxury condominium apartments from 2005 to 2007, while the banking hall became an event space.
  """


def test_overlapped_chunking(
    luminous_control_model: LuminousControlModel, some_large_text: str
) -> None:
    OVERLAP = 8
    MAX_TOKENS = 16

    tracer = InMemoryTracer()
    task = ChunkOverlapTask(
        model=luminous_control_model,
        max_tokens_per_chunk=MAX_TOKENS,
        overlap_length_tokens=OVERLAP,
    )
    output = task.run(ChunkInput(text=some_large_text), tracer)

    tokenizer = luminous_control_model.get_tokenizer()
    output_tokenized = tokenizer.encode_batch(output.chunks)
    for chunk_index in range(len(output_tokenized) - 1):
        first = output_tokenized[chunk_index].tokens
        print(first)

        assert (
            len(first)
            <= MAX_TOKENS + 2
            # `+2` because re-tokenizing the chunk can add a few extra tokens at
            # the beginning or end of each chunk. This is a hack.
        )
        next = output_tokenized[chunk_index + 1].tokens

        found = False
        for offset in range(len(next) - OVERLAP // 2):
            if first[-OVERLAP // 2 :] != next[offset : offset + OVERLAP // 2]:
                continue
            found = True
            break

        assert found
