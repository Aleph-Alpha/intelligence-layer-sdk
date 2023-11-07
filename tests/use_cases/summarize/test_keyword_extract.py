from aleph_alpha_client import Client
import pytest

from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.use_cases.summarize.keyword_extract import KeywordExtract, KeywordExtractInput


@pytest.fixture()
def keyword_extract(client: Client):
    return KeywordExtract(client)

def test_keyword_extract_works(keyword_extract: KeywordExtract):
    input = KeywordExtractInput(chunk="text about computers", language=Language("en"))

    result = keyword_extract.run(input, NoOpDebugLogger())
    assert "computers" in [keyword.lower() for keyword in result] 




