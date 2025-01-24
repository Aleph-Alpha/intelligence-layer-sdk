from unittest.mock import Mock
from uuid import uuid4

from dotenv import load_dotenv
from pytest import fixture

from intelligence_layer.connectors.studio.studio import StudioClient


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
