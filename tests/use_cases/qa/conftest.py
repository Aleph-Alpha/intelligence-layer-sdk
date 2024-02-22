from pytest import fixture

from intelligence_layer.use_cases.qa.single_chunk_qa import SingleChunkQa


@fixture
def single_chunk_qa() -> SingleChunkQa:
    return SingleChunkQa()
