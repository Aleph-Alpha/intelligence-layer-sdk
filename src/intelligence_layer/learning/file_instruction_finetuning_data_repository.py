from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Optional

from fsspec.implementations.local import LocalFileSystem  # type: ignore
from sqlalchemy import ColumnElement, Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from intelligence_layer.evaluation.infrastructure.file_system_based_repository import (
    FileSystemBasedRepository,
)
from intelligence_layer.learning.instruction_finetuning_data_repository import (
    InstructionFinetuningDataRepository,
)
from intelligence_layer.learning.models import (
    Base,
    InstructionFinetuningSample,
    InstructionFinetuningSample_,
)


class FileInstructionFinetuningDataRepository(
    InstructionFinetuningDataRepository, FileSystemBasedRepository
):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)

        self.in_memory_db_synced: bool = False
        self.engine: Engine | None = None
        self.SessionLocal: sessionmaker[Session] | None = None

    def store_sample(self, sample: InstructionFinetuningSample) -> str:
        self.write_utf8(self._sample_path(sample.id), sample.model_dump_json(), True)
        self.in_memory_db_synced = False
        return sample.id

    def store_samples(
        self, samples: Iterable[InstructionFinetuningSample]
    ) -> list[str]:
        sample_ids = list(self.store_sample(sample) for sample in samples)
        self.in_memory_db_synced = False
        return sample_ids

    def head(self, limit: int | None = 100) -> Iterable[InstructionFinetuningSample]:
        file_iter = islice(
            (file for file in self._finetuning_data_path().iterdir() if file.is_file()),
            limit,
        )
        yield from (
            InstructionFinetuningSample.model_validate_json(self.read_utf8(sample_path))
            for sample_path in file_iter
        )

    def sample(self, id: str) -> InstructionFinetuningSample | None:
        sample_path = self._sample_path(id)
        if self.exists(sample_path):
            return InstructionFinetuningSample.model_validate_json(
                self.read_utf8(sample_path)
            )
        return None

    def samples(self, ids: Iterable[str]) -> Iterable[InstructionFinetuningSample]:
        yield from (sample for id in ids if (sample := self.sample(id)))

    def samples_with_filter(
        self, filter_expression: ColumnElement[bool], limit: Optional[int] = 100
    ) -> Iterable[InstructionFinetuningSample]:
        with self._session_local().begin() as session:
            query = session.query(InstructionFinetuningSample_).filter(
                filter_expression
            )
            if limit is not None:
                query = query.limit(limit)

            for db_sample in query.all():
                yield db_sample.to_pydantic()

    def delete_sample(self, id: str) -> None:
        sample_path = self._sample_path(id)
        if self.exists(sample_path):
            self.remove_file(sample_path)
        self.in_memory_db_synced = False

    def delete_samples(self, ids: Iterable[str]) -> None:
        for id in ids:
            self.delete_sample(id)
        self.in_memory_db_synced = False

    def _finetuning_data_path(self) -> Path:
        return self._root_directory / "finetuning_data"

    def _sample_path(self, sample_id: str) -> Path:
        return self._finetuning_data_path() / f"{sample_id}.json"

    def _session_local(self) -> sessionmaker[Session]:
        if self.in_memory_db_synced and self.SessionLocal:
            return self.SessionLocal

        self.engine = create_engine("sqlite:///:memory:", echo=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

        Base.metadata.create_all(self.engine)

        with self.SessionLocal.begin() as session:
            iterator = iter(self.head(limit=None))
            for batch in iter(lambda: list(islice(iterator, 1000)), []):
                db_samples = [
                    InstructionFinetuningSample_.from_pydantic(sample)
                    for sample in batch
                ]
                session.bulk_save_objects(db_samples)

        self.in_memory_db_synced = True
        return self.SessionLocal
