from collections.abc import Iterable
from typing import Generic, Optional, Sequence
from uuid import uuid4

from sqlalchemy import String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.types import ARRAY

from intelligence_layer.connectors.base.json_serializable import (
    JsonSerializable,
    SerializableDict,
)
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
    name: Mapped[str]
    labels: Mapped[list[str]] = mapped_column(ARRAY(String))
    dataset_metadata: Mapped[Optional[str]] = mapped_column(JSONB)
    example_ids: Mapped[list[str]] = mapped_column(ARRAY(String))
    id: Mapped[str] = mapped_column(String, primary_key=True)

   
    def to_dataset(self) -> Dataset:
        if self.metadata:
            metadata: dict[str, JsonSerializable] = dict(self.dataset_metadata)
        else:
            metadata: dict[str, JsonSerializable] = dict()

        return Dataset(
            id=self.id,
            name=self.name,
            labels=set(self.labels),
            metadata=metadata,
        )
    
    @classmethod
    def from_dataset(cls, dataset: Dataset, examples: Sequence[Example]) -> 'SQLDataset':       
        return SQLDataset(
             id=id or str(uuid4()),
                name=dataset.dataset_name,
                labels=list(dataset.labels) if dataset.labels else list(),
                dataset_metadata=JsonSerializer(root=dataset.metadata).model_dump(),
                example_ids=[example.id for example in examples],
            ) 

class SQLExample(Base, Generic[Input, ExpectedOutput]):
    __tablename__ = "examples"
    input: Mapped[Input] = mapped_column(JSONB)
    expected_output: Mapped[ExpectedOutput] = mapped_column(JSONB)
    example_metadata: Mapped[Optional[SerializableDict]] = mapped_column(JSONB)
    id: Mapped[str] = mapped_column(String, primary_key=True)


class SQLAlchemyDatasetRepository(DatasetRepository):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.engine = create_engine(url=url)
        Base.metadata.create_all(self.engine)

    def create_dataset(
        self,
        examples: Iterable[Example[Input, ExpectedOutput]],
        dataset_name: str,
        id: str | None = None,
        labels: set[str] | None = None,
        metadata: SerializableDict | None = None,
    ) -> Dataset:
        if metadata is None:
            metadata = dict()
        if labels is None:
            labels = set()
        dataset_to_persist = Dataset(name=dataset_name, labels=labels, metadata=metadata)
        if id is not None:
            dataset_to_persist.id = id
    
    
        with Session(self.engine) as session:
            for example in examples:
                sql_example = SQLExample(
                    input=JsonSerializer(root=example.input).model_dump(),
                    expected_output=JsonSerializer(
                        root=example.expected_output
                    ).model_dump(),
                    example_metadata=JsonSerializer(root=example.metadata).model_dump(),
                    id=example.id,
                )
                session.add(sql_example)

            sql_dataset = SQLDataset.from_dataset(dataset_to_persist, examples=list(examples))
            session.add(sql_dataset)
            session.commit()

        return dataset_to_persist


    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes a dataset identified by the given dataset ID.

        Args:
            dataset_id: Dataset ID of the dataset to delete.
        """
        pass

    def dataset(self, dataset_id: str) -> Optional[Dataset]:
        with Session(self.engine) as session:
            sql_datasets = session.query(SQLDataset).order_by(SQLDataset.id.asc()).all()

        for sql_dataset in sql_datasets:
            if sql_dataset.id == dataset_id:
                return sql_dataset.to_dataset()

    def dataset_ids(self) -> Iterable[str]:
        with Session(self.engine) as session:
            datasets = session.query(SQLDataset).order_by(SQLDataset.id.asc()).all()

        for dataset in datasets:
            yield dataset.id

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
