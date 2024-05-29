import json
import uuid

from pydantic import BaseModel

from intelligence_layer.core import LogLine


class ParentBaseModel(BaseModel):
    logline: LogLine


class CustomBaseModel(BaseModel):
    id: str


def test_serialize_logline_can_be_read_again() -> None:
    logline = LogLine(trace_id=str(uuid.uuid4()),
                      entry_type="CustomBaseModel",
                      entry=CustomBaseModel(id=str(uuid.uuid4())))

    serialized_logline = logline.model_dump_json()
    logline_as_dict = json.loads(serialized_logline)

    deserialized_logline_without_types = LogLine.model_validate_json(serialized_logline)
    deserialized_logline_without_types.entry = (globals()[deserialized_logline_without_types.entry_type]
                                                .model_validate(logline_as_dict["entry"]))

    assert logline == deserialized_logline_without_types
