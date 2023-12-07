from typing import Iterable, Sequence
from intelligence_layer.connectors import ArgillaClient, Field, Record, ArgillaEvaluation


class StubArgillaClient(ArgillaClient):
    def create_dataset(self, workspace_id: str, dataset_name: str, fields: Sequence[Field]) -> str:
        ...

    def add_record(self, dataset_id: str, record: Record) -> None:
        ...

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        ...


def test_argilla_evaluator_can_evaluate() -> None:
    pass

