import os
from pathlib import Path

from aleph_alpha_client import Client, CompletionRequest, CompletionResponse
from pytest import fixture

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core import NoOpTracer
from intelligence_layer.core.model import LuminousControlModel
from intelligence_layer.use_cases import LongContextSummarizeInput, RecursiveSummarize
from intelligence_layer.use_cases.summarize.steerable_long_context_summarize import (
    SteerableLongContextSummarize,
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
    steerable_long_context_summarize: SteerableLongContextSummarize,
) -> None:
    max_tokens = 1000
    input = LongContextSummarizeInput(text=very_long_text, max_tokens=max_tokens)
    task = RecursiveSummarize(steerable_long_context_summarize)
    output = task.run(input, NoOpTracer())

    assert len(output.summary) < len(very_long_text)
    assert output.generated_tokens < max_tokens
    assert "new orleans" in output.summary.lower()


def test_recursive_summarize_stops_when_num_partial_summaries_stays_same(
    steerable_long_context_summarize: SteerableLongContextSummarize,
) -> None:
    max_tokens = None
    input = LongContextSummarizeInput(text=short_text, max_tokens=max_tokens)
    task = RecursiveSummarize(steerable_long_context_summarize)
    output = task.run(input, NoOpTracer())

    assert output.generated_tokens > 145


def test_recursive_summarize_stops_after_one_chunk(
    recursive_counting_client: RecursiveCountingClient,
) -> None:
    model = LuminousControlModel(
        model="luminous-base-control-20240215", client=recursive_counting_client
    )

    long_context_high_compression_summarize = SteerableLongContextSummarize(
        max_generated_tokens=128, max_tokens_per_chunk=1024, model=model
    )
    input = LongContextSummarizeInput(text=short_text)
    task = RecursiveSummarize(long_context_high_compression_summarize)
    task.run(input, NoOpTracer())

    assert recursive_counting_client.recursive_counter == 1
