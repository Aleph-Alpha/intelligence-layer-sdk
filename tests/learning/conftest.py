import os
from pathlib import Path

from pytest import fixture

from intelligence_layer.core import DetectLanguage, Language, NoOpTracer
from intelligence_layer.core.model import Message
from intelligence_layer.learning import (
    EnrichDomain,
    EnrichQuality,
    FileInstructionFinetuningDataRepository,
    InstructionFinetuningDataHandler,
    InstructionFinetuningSample,
    InstructionFinetuningSampleAttributes,
    PostgresInstructionFinetuningDataRepository,
    RawInstructionFinetuningSample,
)


@fixture
def raw_instruction_finetuning_sample() -> RawInstructionFinetuningSample:
    return RawInstructionFinetuningSample(
        messages=[
            Message(role="user", content="Hi."),
            Message(role="assistant", content="Hello, how can I help you?"),
        ],
        attributes=InstructionFinetuningSampleAttributes(
            source="example", domain="general", quality=5, languages=[Language("en")]
        ),
        external_id="example_1",
    )


@fixture(scope="function")
def instruction_finetuning_sample(
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
) -> InstructionFinetuningSample:
    return InstructionFinetuningSample.from_raw_sample(
        raw_instruction_finetuning_sample
    )


@fixture
def postgres_instruction_finetuning_data_repository() -> (
    PostgresInstructionFinetuningDataRepository
):
    db_user = os.getenv("POSTGRES_USER")
    db_pw = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST")
    db_port = os.getenv("POSTGRES_PORT")

    db_name = os.getenv("POSTGRES_DB")
    db_url = f"postgresql://{db_user}:{db_pw}@{db_host}:{db_port}/{db_name}"

    return PostgresInstructionFinetuningDataRepository(db_url)


@fixture
def file_instruction_finetuning_data_repository(
    tmp_path: Path,
) -> FileInstructionFinetuningDataRepository:
    return FileInstructionFinetuningDataRepository(tmp_path)


@fixture(scope="function")
def instruction_finetuning_data_handler(
    postgres_instruction_finetuning_data_repository: PostgresInstructionFinetuningDataRepository,
) -> InstructionFinetuningDataHandler:
    return InstructionFinetuningDataHandler(
        postgres_instruction_finetuning_data_repository,
        EnrichDomain(["smalltalk", "weather", "gossip"]),
        EnrichQuality(),
        DetectLanguage(),
        [Language("de"), Language("en")],
        Language("en"),
        NoOpTracer(),
    )
