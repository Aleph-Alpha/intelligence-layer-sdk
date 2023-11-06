from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.core.task import Chunk
from intelligence_layer.use_cases.summarize.single_chunk_high_compression_summarize import (
    SingleChunkHighCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.single_chunk_medium_compression_summarize import (
    SingleChunkMediumCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.single_chunk_low_compression_summarize import (
    SingleChunkLowCompressionSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import SingleChunkSummarizeInput

from .test_high_compression_summarize import (
    high_compression_summarize,
)  # fixture import


@fixture
def medium_compression_summarize(
    client: Client,
) -> SingleChunkMediumCompressionSummarize:
    return SingleChunkMediumCompressionSummarize(client)


@fixture
def low_compression_summarize(client: Client) -> SingleChunkLowCompressionSummarize:
    return SingleChunkLowCompressionSummarize(client)


def test_different_compression_summarize(
    high_compression_summarize: SingleChunkHighCompressionSummarize,
    medium_compression_summarize: SingleChunkMediumCompressionSummarize,
    low_compression_summarize: SingleChunkLowCompressionSummarize,
) -> None:
    text = Chunk(
        "The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]"
    )
    input = SingleChunkSummarizeInput(chunk=text, language=Language("en"))
    output_high_compression = high_compression_summarize.run(input, NoOpDebugLogger())
    output_medium_compression = medium_compression_summarize.run(
        input, NoOpDebugLogger()
    )
    output_low_compression = low_compression_summarize.run(input, NoOpDebugLogger())

    assert (
        len(output_high_compression.summary)
        < len(output_medium_compression.summary)
        < len(output_low_compression.summary)
    )
