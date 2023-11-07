from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.use_cases.summarize.long_context_high_compression_summarize import (
    LongContextHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import LongContextSummarizeInput


@fixture
def long_context_high_compression_summarize(
    client: Client,
) -> LongContextHighCompressionSummarize:
    return LongContextHighCompressionSummarize(client)


@fixture
def long_text() -> str:
    return """The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]

While the brown bear's range has shrunk, and it has faced local extinctions across its wide range, it remains listed as a least concern species by the International Union for Conservation of Nature (IUCN) with a total estimated population in 2017 of 110,000. As of 2012, this and the American black bear are the only bear species not classified as threatened by the IUCN, though the large sizes of both bears may be a disadvantage due to increased competition with humans.[1][3][7] Populations that were hunted to extinction in the 19th and 20th centuries are the Atlas bear of North Africa and the Californian, Ungavan[11][12] and Mexican populations of the grizzly bear of North America. Many of the populations in the southern parts of Eurasia are highly endangered as well.[1][13] One of the smaller-bodied forms, the Himalayan brown bear, is critically endangered, occupying only 2% of its former range and threatened by uncontrolled poaching for its body parts.[14] The Marsican brown bear of central Italy is one of several currently isolated populations of the Eurasian brown bear and is believed to have a population of just 50 to 60 bears.[10][15]

Evolution and taxonomy
The brown bear is sometimes referred to as the bruin, from Middle English. This name originated in the fable History of Reynard the Fox translated by William Caxton from Middle Dutch bruun or bruyn, meaning brown (the color).[16][17] In the mid-19th century United States, the brown bear was termed "Old Ephraim" and sometimes as "Moccasin Joe".[18]

The scientific name of the brown bear, Ursus arctos, comes from the Latin ursus, meaning "bear",[19] and the Greek ἄρκτος/arktos, also meaning "bear".[20]

Generalized names and evolution
Brown bears are thought to have evolved from Ursus etruscus in Asia.[21][22] The brown bear, per Kurten (1976), has been stated as "clearly derived from the Asian population of Ursus savini about 800,000 years ago; spread into Europe, to the New World."[23] A genetic analysis indicated that the brown bear lineage diverged from the cave bear species complex approximately 1.2–1.4 million years ago, but did not clarify if U. savini persisted as a paraspecies for the brown bear before perishing.[24] The oldest fossils positively identified as from this species occur in China from about 0.5 million years ago. Brown bears entered Europe about 250,000 years ago and North Africa shortly after.[21][25] Brown bear remains from the Pleistocene period are common in the British Isles, where it is thought they might have outcompeted cave bears (Ursus spelaeus). The species entered Alaska 100,000 years ago, though they did not move south until 13,000 years ago.[21] It is speculated that brown bears were unable to migrate south until the extinction of the much larger giant short-faced bear (Arctodus simus).[26][27]

Several paleontologists suggest the possibility of two separate brown bear migrations. First, the inland brown bears, also known as grizzlies, are thought to stem from narrow-skulled bears which migrated from northern Siberia to central Alaska and the rest of the continent. Moreover, the Kodiak bears descend from broad-skulled bears from Kamchatka, which colonized the Alaskan peninsula. Brown bear fossils discovered in Ontario, Ohio, Kentucky and Labrador show that the species occurred farther east than indicated in historic records.[21] In North America, two types of the subspecies Ursus arctos horribilis are generally recognized—the coastal brown bear and the inland grizzly bear; these two types broadly define the range of sizes of all brown bear subspecies.[13]

Scientific taxonomy
Main article: Subspecies of brown bear

Adult female Eurasian brown bear, the nominate subspecies
There are many methods used by scientists to define bear species and subspecies, as no one method is always effective. Brown bear taxonomy and subspecies classification has been described as "formidable and confusing," with few authorities listing the same specific set of subspecies.[28] Genetic testing is now perhaps the most important way to scientifically define brown bear relationships and names. Generally, genetic testing uses the word clade rather than species because a genetic test alone cannot define a biological species. Most genetic studies report on how closely related the bears are (or their genetic distance). There are hundreds of obsolete brown bear subspecies, each with its own name, so this can become confusing. Hall (1981) lists 86 different types, and even as many as 90 have been proposed.[29][30] However, recent DNA analysis has identified as few as five main clades which contain all extant brown bears,[31][32] while a 2017 phylogenetic study revealed nine clades, including one representing polar bears.[33] As of 2005, 15 extant or recently extinct subspecies were recognized by the general scientific community.[34][35]

As well as the exact number of overall brown bear subspecies, its precise relationship to the polar bear also remains in debate. The polar bear is a recent offshoot of the brown bear. The point at which the polar bear diverged from the brown bear is unclear, with estimations based on genetics and fossils ranging from 400,000 to 70,000 years ago, but most recent analysis has indicated that the polar bear split somewhere between 275,000 and 150,000 years ago.[36] Under some definitions, the brown bear can be construed as the paraspecies for the polar bear.[37][38][39][40]

DNA analysis shows that, apart from recent human-caused population fragmentation,[41] brown bears in North America are generally part of a single interconnected population system, with the exception of the population (or subspecies) in the Kodiak Archipelago, which has probably been isolated since the end of the last Ice Age.[42][43] These data demonstrate that U. a. gyas, U. a. horribilis, U. a. sitkensis and U. a. stikeenensis are not distinct or cohesive groups, and would more accurately be described as ecotypes. For example, brown bears in any particular region of the Alaska coast are more closely related to adjacent grizzly bears than to distant populations of brown bears,[44] the morphological distinction seemingly driven by brown bears having access to a rich salmon food source, while grizzly bears live at higher elevation, or further from the coast, where plant material is the base of the diet. The history of the bears of the Alexander Archipelago is unusual in that these island populations carry polar bear DNA, presumably originating from a population of polar bears that was left behind at the end of the Pleistocene, but have since been connected with adjacent mainland populations through movement of males, to the point where their nuclear genomes are now more than 90% of brown bear ancestry.[45]

Brown bears are apparently divided into five different clades, some of which coexist or co-occur in different regions.[3]"""


def test_long_context_high_compression_summarize_en(
    long_context_high_compression_summarize: LongContextHighCompressionSummarize,
    long_text: str,
) -> None:
    input = LongContextSummarizeInput(text=long_text)
    output = long_context_high_compression_summarize.run(input, NoOpDebugLogger())

    assert output.partial_summaries
    assert any(
        "bear" in partial_summary.summary
        for partial_summary in output.partial_summaries
    )
    assert len(
        " ".join(
            partial_summary.summary for partial_summary in output.partial_summaries
        )
    ) < len(long_text)
