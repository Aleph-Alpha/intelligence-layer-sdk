from collections.abc import Iterable
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from intelligence_layer.connectors.base.json_serializable import SerializableDict
from intelligence_layer.core.task import Input
from intelligence_layer.core.tracer.tracer import JsonSerializer
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import (
    Dataset,
    Example,
    ExpectedOutput,
)


class Base(DeclarativeBase):
    pass


# Json is supported by SQLAlchemy. But mutability is difficult to handle. See:
# https://amercader.net/blog/beware-of-json-fields-in-sqlalchemy/


class SQLDataset(Base):
    __tablename__ = "datasets"
    name = Column(str)
    labels = Column(list[str])
    metadata = Column(JSONB)
    example_ids = Column(list[str])
    id = Column(str, primary_key=True)


class SQLExample(Base):
    __tablename__ = "examples"
    input = Column(JSONB)
    expected_output = Column(JSONB)
    metadata = Column(JSONB)
    id = Column(str, primary_key=True)


class SQLAlchemyDatasetRepository(DatasetRepository):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.engine = create_engine(url=url)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
        dataset_name: str,
        id: str | None = None,
        labels: set[str] | None = None,
        metadata: SerializableDict | None = None,
    ) -> Dataset:
        with self.session() as session:
            for example in examples:
                sql_example = SQLExample(
                    input=JsonSerializer(root=example.input).model_dump(),
                    expected_output=JsonSerializer(
                        root=example.expected_output
                    ).model_dump(),
                    metadata=JsonSerializer(root=example.metadata).model_dump(),
                    id=example.id,
                )
                session.add(sql_example)
            sql_dataset = SQLDataset(
                id=id or str(uuid4()),
                name=dataset_name,
                labels=labels,
                metadata=JsonSerializer(root=metadata).model_dump(),
                example_ids=[example.id for example in examples],
            )
            session.add(sql_dataset)
            session.commit()

    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.
        """
        pass

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Returns a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.

        Returns:
            :class:`Dataset` if it was not, `None` otherwise.
        """
        pass

    def dataset_ids(self) -> Iterable[str]:
        """Returns all sorted dataset IDs.

        Returns:
            :class:`Iterable` of dataset IDs.
        """
        pass

    def example(
        self,
        dataset_id: str,
        example_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Optional[Example[Input, ExpectedOutput]]:
        """Returns an :class:`Example` for the given dataset ID and example ID.

        Args:
            dataset_id: Dataset ID of the linked dataset.
            example_id: ID of the example to retrieve.
            input_type: Input type of the example.
            expected_output_type: Expected output type of the example.

        Returns:
            :class:`Example` if it was found, `None` otherwise.
        """
        pass

    def examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
        examples_to_skip: Optional[frozenset[str]] = None,
    ) -> Iterable[Example[Input, ExpectedOutput]]:
        """Returns all :class:`Example`s for the given dataset ID sorted by their ID.

        Args:
            dataset_id: Dataset ID whose examples should be retrieved.
            input_type: Input type of the example.
            expected_output_type: Expected output type of the example.
            examples_to_skip: Optional list of example IDs. Those examples will be excluded from the output.

        Returns:
            :class:`Iterable` of :class`Example`s.
        """
        pass
