from collections.abc import Sequence
from unittest.mock import Mock
from uuid import uuid4

from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.connectors.studio.studio import StudioClient, StudioExample


@fixture
def studio_client() -> StudioClient:
    load_dotenv()
    project_name = str(uuid4())
    client = StudioClient(project_name)
    client.create_project(project_name)
    return client


@fixture
def mock_studio_client() -> Mock:
    return Mock(spec=StudioClient)


@fixture
def examples() -> Sequence[StudioExample[str, str]]:
    return [
        StudioExample(input="input_str", expected_output="output_str"),
        StudioExample(input="input_str2", expected_output="output_str2"),
    ]
