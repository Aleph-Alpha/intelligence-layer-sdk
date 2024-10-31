from collections.abc import Iterable
from itertools import islice
from typing import Optional

from sqlalchemy import ColumnElement, create_engine
from sqlalchemy.orm import Query, sessionmaker

from intelligence_layer.learning.instruction_finetuning_data_repository import (
    InstructionFinetuningDataRepository,
)
from intelligence_layer.learning.models import (
    Base,
    InstructionFinetuningSample,
    InstructionFinetuningSample_,
)


class PostgresInstructionFinetuningDataRepository(InstructionFinetuningDataRepository):
    def __init__(
        self,
        database_url: str,
        page_size: int = 100,
    ) -> None:
        self.engine = create_engine(database_url, pool_size=20, max_overflow=0)
        self.Session = sessionmaker(bind=self.engine)

        Base.metadata.create_all(self.engine)

        self.page_size = page_size

    def store_sample(self, sample: InstructionFinetuningSample) -> str:
        with self.Session.begin() as session:
            db_sample = InstructionFinetuningSample_.from_pydantic(sample)
            session.merge(db_sample)
        return sample.id

    def store_samples(
        self, samples: Iterable[InstructionFinetuningSample]
    ) -> list[str]:
        stored_ids: list[str] = []
        with self.Session.begin() as session:
            iterator = iter(samples)
            for batch in iter(lambda: list(islice(iterator, 1000)), []):
                db_samples = [
                    InstructionFinetuningSample_.from_pydantic(sample)
                    for sample in batch
                ]
                session.bulk_save_objects(db_samples)
                stored_ids.extend(sample.id for sample in batch)

        return stored_ids

    def head(self, limit: Optional[int] = 100) -> Iterable[InstructionFinetuningSample]:
        retrieved, page = 0, 1

        with self.Session.begin() as session:
            while retrieved < limit if limit is not None else True:
                offset = (page - 1) * self.page_size
                query = session.query(InstructionFinetuningSample_)
                query = query.offset(offset).limit(
                    min(self.page_size, limit - retrieved)
                    if limit is not None
                    else self.page_size
                )
                db_samples = query.all()

                if not db_samples:
                    break

                for db_sample in db_samples:
                    yield db_sample.to_pydantic()
                    retrieved += 1

                if limit is not None and retrieved >= limit:
                    break

                page += 1

    def sample(self, id: str) -> Optional[InstructionFinetuningSample]:
        with self.Session.begin() as session:
            db_sample = (
                session.query(InstructionFinetuningSample_).filter_by(id=id).first()
            )
            return db_sample.to_pydantic() if db_sample else None

    def samples(self, ids: Iterable[str]) -> Iterable[InstructionFinetuningSample]:
        page = 1

        with self.Session.begin() as session:
            while True:
                offset = (page - 1) * self.page_size
                db_samples = (
                    session.query(InstructionFinetuningSample_)
                    .filter(InstructionFinetuningSample_.id.in_(ids))
                    .offset(offset)
                    .limit(self.page_size)
                    .all()
                )

                if not db_samples:
                    break

                for db_sample in db_samples:
                    yield db_sample.to_pydantic()

                page += 1

    def samples_with_filter(
        self, filter_expression: ColumnElement[bool], limit: Optional[int] = 100
    ) -> Iterable[InstructionFinetuningSample]:
        with self.Session() as session:
            query = session.query(InstructionFinetuningSample_).filter(
                filter_expression
            )
            query = self._set_query_limit(query, limit)
            for db_sample in query.all():
                yield db_sample.to_pydantic()

    def delete_sample(self, id: str) -> None:
        with self.Session.begin() as session:
            session.query(InstructionFinetuningSample_).filter_by(id=id).delete()

    def delete_samples(self, ids: Iterable[str]) -> None:
        with self.Session.begin() as session:
            session.query(InstructionFinetuningSample_).filter(
                InstructionFinetuningSample_.id.in_(ids)
            ).delete(synchronize_session=False)

    @staticmethod
    def _set_query_limit(
        query: Query[InstructionFinetuningSample_], limit: Optional[int]
    ) -> Query[InstructionFinetuningSample_]:
        if limit is not None:
            return query.limit(limit)
        return query
