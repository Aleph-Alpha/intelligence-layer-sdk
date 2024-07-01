from collections.abc import Iterable

from intelligence_layer.evaluation.dataset.domain import Example
from intelligence_layer.evaluation.dataset.sql_alchemy_dataset_repository import (
    SQLAlchemyDatasetRepository,
)


def test_save_examples(sequence_examples: Iterable[Example[str, None]]):
    dataset_repo = SQLAlchemyDatasetRepository(url="postgresql://il_sdk:test@localhost:5433/postgres")
    dataset_repo.create_dataset(sequence_examples, "test_dataset")
