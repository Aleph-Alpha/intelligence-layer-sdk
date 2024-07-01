from collections.abc import Iterable
from typing import Callable

from pytest import fixture

from intelligence_layer.evaluation.dataset.domain import Example
from intelligence_layer.evaluation.dataset.sql_alchemy_dataset_repository import (
    Base,
    SQLAlchemyDatasetRepository,
)

@fixture
def sql_alchemy_dataset_repo() -> SQLAlchemyDatasetRepository:
    repo = SQLAlchemyDatasetRepository(
        url="postgresql://il_sdk:test@localhost:5433/postgres"
    )
    Base.metadata.drop_all(bind=repo.engine)
    Base.metadata.create_all(bind=repo.engine)
    return repo

def test_save_examples(
        sql_alchemy_dataset_repo: SQLAlchemyDatasetRepository,
        sequence_examples: Iterable[Example[str, None]]
) -> None:
    sql_alchemy_dataset_repo.create_dataset(sequence_examples, "test_dataset")
