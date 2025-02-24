from collections.abc import Sequence
from unittest.mock import Mock
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel
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


class PydanticType(BaseModel):
    data: int


@fixture
def examples() -> Sequence[StudioExample[PydanticType, PydanticType]]:
    return [
        StudioExample[PydanticType, PydanticType](
            input=PydanticType(data=i), expected_output=PydanticType(data=i)
        )
        for i in range(2)
    ]


@fixture
def many_examples() -> Sequence[StudioExample[PydanticType, PydanticType]]:
    examples = [
        StudioExample[PydanticType, PydanticType](
            input=PydanticType(data=i), expected_output=PydanticType(data=i)
        )
        for i in range(15)
    ]
    return examples
