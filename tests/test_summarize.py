from aleph_alpha_client import Client
from pytest import fixture
from intelligence_layer.summarize import (
    SummarizeInput,
    ShortBodySummarize,
)

@fixture
def qa(client: Client) -> ShortBodySummarize:
    return ShortBodySummarize(client, "info")


def test_summarize(summarize: ShortBodySummarize) -> None:
    input = SummarizeInput(
        text="""The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]

While the brown bear's range has shrunk, and it has faced local extinctions across its wide range, it remains listed as a least concern species by the International Union for Conservation of Nature (IUCN) with a total estimated population in 2017 of 110,000. As of 2012, this and the American black bear are the only bear species not classified as threatened by the IUCN, though the large sizes of both bears may be a disadvantage due to increased competition with humans.[1][3][7] Populations that were hunted to extinction in the 19th and 20th centuries are the Atlas bear of North Africa and the Californian, Ungavan[11][12] and Mexican populations of the grizzly bear of North America. Many of the populations in the southern parts of Eurasia are highly endangered as well.[1][13] One of the smaller-bodied forms, the Himalayan brown bear, is critically endangered, occupying only 2% of its former range and threatened by uncontrolled poaching for its body parts.[14] The Marsican brown bear of central Italy is one of several currently isolated populations of the Eurasian brown bear and is believed to have a population of just 50 to 60 bears."""
    )
    output = summarize.run(input)

    assert output.summary
    assert "bear" in output.summary.lower
