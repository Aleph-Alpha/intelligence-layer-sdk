from pytest import fixture

from intelligence_layer.core import LuminousControlModel
from intelligence_layer.use_cases import SingleChunkQa


@fixture
def single_chunk_qa(luminous_control_model: LuminousControlModel) -> SingleChunkQa:
    return SingleChunkQa(luminous_control_model)
