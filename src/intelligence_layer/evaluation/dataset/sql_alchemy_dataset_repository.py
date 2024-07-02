import json
from collections.abc import Iterable, Sequence
from typing import Generic, Optional
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
        metadata: dict[str, JsonSerializable] = dict()
        if self.dataset_metadata:
            metadata = json.loads(self.dataset_metadata)

        return Dataset(
            id=self.id,
            name=self.name,
            labels=set(self.labels),
            metadata=metadata,
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset, example_ids: Sequence[str]) -> "SQLDataset":
        return SQLDataset(
            id=dataset.id or str(uuid4()),
            name=dataset.name,
            labels=list(dataset.labels) if dataset.labels else list(),
            dataset_metadata=json.dumps(dataset.metadata),
            example_ids=example_ids,
        )


class SQLExample(Base, Generic[Input, ExpectedOutput]):
    __tablename__ = "examples"
    input: Mapped[Input] = mapped_column(JSONB)
    expected_output: Mapped[ExpectedOutput] = mapped_column(JSONB)
    example_metadata: Mapped[Optional[SerializableDict]] = mapped_column(JSONB)
    id: Mapped[str] = mapped_column(String, primary_key=True)

    @classmethod
    def from_example(
        cls, example: Example[Input, ExpectedOutput]
    ) -> "SQLExample[Input, ExpectedOutput]":
        return SQLExample(
            input=JsonSerializer(root=example.input).model_dump(),
            expected_output=JsonSerializer(root=example.expected_output).model_dump(),
            example_metadata=json.dumps(example.metadata),
            id=example.id,
        )

    def to_example(
        self,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Example[Input, ExpectedOutput]:
        metadata: dict[str, JsonSerializable] = dict()
        if self.example_metadata:
            metadata = json.loads(self.example_metadata)
        example = Example[input_type, expected_output_type](  # type: ignore
            input=input_type.model_validate_json(self.input),
            expected_output=expected_output_type.model_validate_json(
                self.expected_output
            ),
            id=self.id,
            metadata=metadata,
        )
        return example


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
        dataset_to_persist = Dataset(
            name=dataset_name, labels=labels, metadata=metadata
        )
        if id is not None:
            dataset_to_persist.id = id

        with Session(self.engine) as session:
            for example in examples:
                sql_example = SQLExample.from_example(example)
                session.add(sql_example)

            sql_dataset = SQLDataset.from_dataset(
                dataset_to_persist, example_ids=list(example.id for example in examples)
            )
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
            sql_dataset = (
                session.query(SQLDataset)
                .filter_by(id=dataset_id)
                .order_by(SQLDataset.id.asc())
                .one_or_none()
            )
        if sql_dataset is None:
            return None

        return sql_dataset.to_dataset()

    def dataset_ids(self) -> Iterable[str]:
        with Session(self.engine) as session:
            datasets = session.query(SQLDataset).order_by(SQLDataset.id.asc()).all()

        for dataset in datasets:
            yield dataset.id

        return None

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
        with Session(self.engine) as session:
            sql_dataset = (
                session.query(SQLDataset)
                .filter_by(id=dataset_id)
                .order_by(SQLDataset.id.asc())
                .one_or_none()
            )

            sql_examples = (
                session.query(SQLExample)
                .filter(SQLExample.id.in_(sql_dataset.example_ids))
                .all()
            )

        for sql_example in sql_examples:
            yield sql_example.to_example(input_type, expected_output_type)
