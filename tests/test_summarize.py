from aleph_alpha_client import Client
from pytest import fixture
from intelligence_layer.summarize import (
    SummarizeInput,
    ShortBodySummarize,
)

from intelligence_layer.task import JsonDebugLogger


@fixture
def summarize(client: Client) -> ShortBodySummarize:
    return ShortBodySummarize(client)


def test_summarize(summarize: ShortBodySummarize) -> None:
    text = """The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]"""
    input = SummarizeInput(text=text)
    output = summarize.run(input, JsonDebugLogger(name="summarize"))

    assert output.summary
    assert "bear" in output.summary.lower()
    assert len(output.summary) < len(text)
    assert output.highlights
