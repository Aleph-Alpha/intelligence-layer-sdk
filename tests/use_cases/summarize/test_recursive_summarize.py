import os
from pathlib import Path

from aleph_alpha_client import Client, CompletionRequest, CompletionResponse
from pytest import fixture

from intelligence_layer.core.tracer import NoOpTracer
from intelligence_layer.use_cases.summarize.long_context_high_compression_summarize import (
    LongContextHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.recursive_summarize import (
    RecursiveSummarize,
    RecursiveSummarizeInput,
)


class RecursiveCountingClient(Client):
    recursive_counter: int = 0

    def complete(self, request: CompletionRequest, model: str) -> CompletionResponse:
        self.recursive_counter += 1
        return super().complete(request, model)


short_text = """The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]"""


@fixture
def recursive_counting_client() -> RecursiveCountingClient:
    aa_token = os.getenv("AA_TOKEN")
    assert aa_token
    return RecursiveCountingClient(aa_token)


@fixture
def very_long_text() -> str:
    with open(
        Path(__file__).parent / "very_long_text.txt", "r", encoding="utf-8"
    ) as file:
        return file.read()


def test_recursive_summarize_stops_when_hitting_max_tokens(
    very_long_text: str,
    long_context_high_compression_summarize: LongContextHighCompressionSummarize,
) -> None:
    max_tokens = 1000
    input = RecursiveSummarizeInput(text=very_long_text, max_tokens=max_tokens)
    task = RecursiveSummarize(long_context_high_compression_summarize)
    output = task.run(input, NoOpTracer())

    assert len(output.summary) < len(very_long_text)
    assert output.generated_tokens < max_tokens
    assert "new orleans" in output.summary.lower()


def test_recursive_summarize_stops_when_hitting_max_loops(
    very_long_text: str,
    recursive_counting_client: RecursiveCountingClient,
) -> None:
    long_context_high_compression_summarize = LongContextHighCompressionSummarize(
        recursive_counting_client, model="luminous-base"
    )
    input = RecursiveSummarizeInput(text=very_long_text, max_loops=1)
    task = RecursiveSummarize(long_context_high_compression_summarize)
    output = task.run(input, NoOpTracer())

    assert len(output.summary) < len(very_long_text)
    assert (
        recursive_counting_client.recursive_counter == 71
    )  # text is chunked into 71 chunks
    assert "new orleans" in output.summary.lower()


def test_recursive_summarize_stops_after_one_chunk(
    recursive_counting_client: RecursiveCountingClient,
) -> None:
    long_context_high_compression_summarize = LongContextHighCompressionSummarize(
        recursive_counting_client, model="luminous-base"
    )
    input = RecursiveSummarizeInput(text=short_text)
    task = RecursiveSummarize(long_context_high_compression_summarize)
    task.run(input, NoOpTracer())

    assert recursive_counting_client.recursive_counter == 1
