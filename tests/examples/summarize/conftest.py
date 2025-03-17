from pytest import fixture

from intelligence_layer.core import Chunk, LuminousControlModel, TextChunk
from intelligence_layer.examples.summarize.steerable_long_context_summarize import (
    SteerableLongContextSummarize,
)
from intelligence_layer.examples.summarize.steerable_single_chunk_summarize import (
    SteerableSingleChunkSummarize,
)


@fixture
def steerable_single_chunk_summarize(
    llama_control_model: LuminousControlModel,
) -> SteerableSingleChunkSummarize:
    return SteerableSingleChunkSummarize(llama_control_model)


@fixture
def chunk() -> TextChunk:
    return TextChunk(
        "The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]"
    )


@fixture
def steerable_long_context_summarize(
    llama_control_model: LuminousControlModel,
) -> SteerableLongContextSummarize:
    return SteerableLongContextSummarize(
        summarize=SteerableSingleChunkSummarize(
            llama_control_model, max_generated_tokens=128
        ),
        chunk=Chunk(llama_control_model, max_tokens_per_chunk=1024),
    )


@fixture
def long_text() -> str:
    return """The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]

While the brown bear's range has shrunk, and it has faced local extinctions across its wide range, it remains listed as a least concern species by the International Union for Conservation of Nature (IUCN) with a total estimated population in 2017 of 110,000. As of 2012, this and the American black bear are the only bear species not classified as threatened by the IUCN, though the large sizes of both bears may be a disadvantage due to increased competition with humans.[1][3][7] Populations that were hunted to extinction in the 19th and 20th centuries are the Atlas bear of North Africa and the Californian, Ungavan[11][12] and Mexican populations of the grizzly bear of North America. Many of the populations in the southern parts of Eurasia are highly endangered as well.[1][13] One of the smaller-bodied forms, the Himalayan brown bear, is critically endangered, occupying only 2% of its former range and threatened by uncontrolled poaching for its body parts.[14] The Marsican brown bear of central Italy is one of several currently isolated populations of the Eurasian brown bear and is believed to have a population of just 50 to 60 bears.[10][15]

Evolution and taxonomy
The brown bear is sometimes referred to as the bruin, from Middle English. This name originated in the fable History of Reynard the Fox translated by William Caxton from Middle Dutch bruun or bruyn, meaning brown (the color).[16][17] In the mid-19th century United States, the brown bear was termed "Old Ephraim" and sometimes as "Moccasin Joe".[18]

The scientific name of the brown bear, Ursus arctos, comes from the Latin ursus, meaning "bear",[19] and the Greek ἄρκτος/arktos, also meaning "bear".[20]

Generalized names and evolution
Brown bears are thought to have evolved from Ursus etruscus in Asia.[21][22] The brown bear, per Kurten (1976), has been stated as "clearly derived from the Asian population of Ursus savini about 800,000 years ago; spread into Europe, to the New World."[23] A genetic analysis indicated that the brown bear lineage diverged from the cave bear species complex approximately 1.2–1.4 million years ago, but did not clarify if U. savini persisted as a paraspecies for the brown bear before perishing.[24] The oldest fossils positively identified as from this species occur in China from about 0.5 million years ago. Brown bears entered Europe about 250,000 years ago and North Africa shortly after.[21][25] Brown bear remains from the Pleistocene period are common in the British Isles, where it is thought they might have outcompeted cave bears (Ursus spelaeus). The species entered Alaska 100,000 years ago, though they did not move south until 13,000 years ago.[21] It is speculated that brown bears were unable to migrate south until the extinction of the much larger giant short-faced bear (Arctodus simus).[26][27]

Several paleontologists suggest the possibility of two separate brown bear migrations. First, the inland brown bears, also known as grizzlies, are thought to stem from narrow-skulled bears which migrated from northern Siberia to central Alaska and the rest of the continent. Moreover, the Kodiak bears descend from broad-skulled bears from Kamchatka, which colonized the Alaskan peninsula. Brown bear fossils discovered in Ontario, Ohio, Kentucky and Labrador show that the species occurred farther east than indicated in historic records.[21] In North America, two types of the subspecies Ursus arctos horribilis are generally recognized—the coastal brown bear and the inland grizzly bear; these two types broadly define the range of sizes of all brown bear subspecies.[13]"""
