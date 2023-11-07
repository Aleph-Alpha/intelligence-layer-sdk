import pytest
from aleph_alpha_client import Client

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language, LanguageNotSupportedError
from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.use_cases.summarize.keyword_extract import (
    KeywordExtract,
    KeywordExtractInput,
)


@pytest.fixture()
def keyword_extract(client: Client) -> KeywordExtract:
    return KeywordExtract(client)


def test_keyword_extract_works(keyword_extract: KeywordExtract) -> None:
    input = KeywordExtractInput(
        chunk=Chunk("text about computers"), language=Language("en")
    )

    result = keyword_extract.run(input, NoOpDebugLogger())
    assert "computers" in [keyword.lower() for keyword in result]


def test_keyword_extract_raises_for_unsupported_language(
    keyword_extract: KeywordExtract,
) -> None:
    input = KeywordExtractInput(
        chunk=Chunk("text about computers"), language=Language("pt")
    )
    with pytest.raises(LanguageNotSupportedError) as _:
        keyword_extract.run(input, NoOpDebugLogger())
