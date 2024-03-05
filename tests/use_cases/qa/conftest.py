from pytest import fixture

from intelligence_layer.core import LuminousControlModel
from intelligence_layer.use_cases import MultipleChunkQa, SingleChunkQa


@fixture
def single_chunk_qa(luminous_control_model: LuminousControlModel) -> SingleChunkQa:
    return SingleChunkQa(luminous_control_model)


@fixture
def multiple_chunk_qa(single_chunk_qa: SingleChunkQa) -> MultipleChunkQa:
    return MultipleChunkQa(single_chunk_qa)
