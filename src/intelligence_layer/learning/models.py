from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import DeclarativeBase, mapped_column

from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.model import FinetuningMessage, Message


class TripletTransformation(Enum):
    INSTRUCTION_AS_SYSTEM = "instruction as system"
    MERGE_USER_AND_SYSTEM = "merge user and system"


class InstructionFinetuningSampleAttributes(BaseModel, frozen=True):
    source: str
    domain: str | None = None
    quality: int | None = None
    languages: Sequence[Language] | None = None


class RawInstructionFinetuningSample(BaseModel, frozen=True):
    messages: Sequence[Message]
    attributes: InstructionFinetuningSampleAttributes
    external_id: Optional[str] = None


class InstructionFinetuningSample(RawInstructionFinetuningSample, frozen=True):
    id: str = Field(default_factory=lambda: str(uuid4()))

    @classmethod
    def from_raw_sample(
        cls, raw_sample: RawInstructionFinetuningSample
    ) -> "InstructionFinetuningSample":
        return InstructionFinetuningSample(
            messages=raw_sample.messages,
            attributes=raw_sample.attributes,
            external_id=raw_sample.external_id,
        )

    @classmethod
    def from_chat_messages_json(
        cls, chat_messages_json: Mapping[Any, Any], source: str
    ) -> "InstructionFinetuningSample":
        return InstructionFinetuningSample(
            messages=chat_messages_json["messages"],
            attributes=InstructionFinetuningSampleAttributes(
                source=source,
                domain=chat_messages_json.get("domain"),
                quality=chat_messages_json.get("quality"),
                languages=chat_messages_json.get("languages"),
            ),
            external_id=chat_messages_json.get("id")
            or chat_messages_json.get("external_id"),
        )

    @classmethod
    def from_triplet_json(
        cls,
        triplet_json: Mapping[Any, Any],
        source: str,
        triplet_transformation: TripletTransformation,
    ) -> "InstructionFinetuningSample":
        instruction, input, output = (
            triplet_json["triplet"].get("instruction"),
            triplet_json["triplet"].get("input"),
            triplet_json["triplet"].get("output"),
        )

        if (
            triplet_transformation == TripletTransformation.INSTRUCTION_AS_SYSTEM
            and input
        ):
            messages = [
                Message(role="system", content=instruction),
                Message(role="user", content=input),
            ]
        else:
            messages = [
                Message(
                    role="user",
                    content=f"{instruction}\n\n{input}" if input else instruction,
                )
            ]

        messages.append(Message(role="assistant", content=output))

        return InstructionFinetuningSample(
            messages=messages,
            attributes=InstructionFinetuningSampleAttributes(
                source=source,
                domain=triplet_json.get("domain"),
                quality=triplet_json.get("quality"),
                languages=triplet_json.get("languages"),
            ),
            external_id=triplet_json.get("id") or triplet_json.get("external_id"),
        )


class InvalidSampleError(ValueError):
    pass


class Base(DeclarativeBase):
    pass


class InstructionFinetuningSample_(Base):
    __tablename__ = "instruction_finetuning_samples"

    id = mapped_column(String, primary_key=True)
    messages = mapped_column(JSON, nullable=False)
    source = mapped_column(String, nullable=False)
    domain = mapped_column(String)
    quality = mapped_column(Integer)
    languages = mapped_column(JSON)
    external_id = mapped_column(String)

    @classmethod
    def from_pydantic(
        cls, pydantic_obj: InstructionFinetuningSample
    ) -> "InstructionFinetuningSample_":
        return cls(
            id=pydantic_obj.id,
            messages=[message.model_dump() for message in pydantic_obj.messages],
            source=pydantic_obj.attributes.source,
            domain=pydantic_obj.attributes.domain,
            quality=pydantic_obj.attributes.quality,
            languages=[lang.iso_639_1 for lang in pydantic_obj.attributes.languages]
            if pydantic_obj.attributes.languages
            else None,
            external_id=pydantic_obj.external_id,
        )

    def to_pydantic(self) -> InstructionFinetuningSample:
        return InstructionFinetuningSample(
            id=self.id,
            messages=[Message(**message) for message in self.messages],
            attributes=InstructionFinetuningSampleAttributes(
                source=self.source,
                domain=self.domain,
                quality=self.quality,
                languages=[Language(lang) for lang in self.languages]
                if self.languages
                else None,
            ),
            external_id=self.external_id,
        )


class TrainSet(BaseModel):
    data: Sequence[Sequence[FinetuningMessage]]
    ids: Sequence[str]
    statistics: Mapping[str, Mapping[Any, int]]
