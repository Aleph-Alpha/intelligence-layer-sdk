from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.use_cases.qa.single_chunk_qa import SingleChunkQa


from pytest import fixture


@fixture
def single_chunk_qa(client: AlephAlphaClientProtocol) -> SingleChunkQa:
    return SingleChunkQa(client)
