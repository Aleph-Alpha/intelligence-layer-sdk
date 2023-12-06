from typing import Sequence
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core.evaluation.argilla_evaluator import ArgillaClient
from intelligence_layer.core.evaluation.domain import Example


class DummyInput(BaseModel):
    query: str


class DummyOutput(BaseModel):
    answer: str


ExpectedOutput = str


@fixture
def argilla_client() -> ArgillaClient:
    return ArgillaClient()


def test_argilla_client_allows_dataset_creation(argilla_client: ArgillaClient) -> None:
    upload: Sequence[tuple[Example[DummyInput, ExpectedOutput], DummyOutput]] = [
        (
            Example(input=DummyInput(query="Was ist 1+1?"), expected_output="2"),
            DummyOutput(answer="3")
        ),
        (
            Example(input=DummyInput(query="Wirklich?"), expected_output="nein"),
            DummyOutput(answer="ja")
        )
    ]
    argilla_client.upload(upload)
