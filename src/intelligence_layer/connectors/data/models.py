import io
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel, to_snake


class BaseDataModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True)


class MediaTypes(str, Enum):
    jsonlines = "application/x-ndjson"
    jsonlines_ = "application/jsonlines"


class Modality(str, Enum):
    text = "text"


class DataRepository(BaseDataModel):
    """Data Repository model.

    Attributes:
    repository_id: Repository ID that identifies the repository(group of datasets)
    name: Name of the repository
    mutable: Indicates if the datasets in the repository are mutable or not
    media_type: Media type of the data: application/json, application/csv, etc.
    modality: Modality of the data: image, text, etc.
    created_at: Datetime when the repository was created
    updated_at: Datetime when the repository was updated
    """

    repository_id: str
    name: str
    mutable: bool
    media_type: MediaTypes
    modality: Modality
    created_at: datetime
    updated_at: datetime


class DataRepositoryCreate(BaseDataModel):
    """Data Repository creation model.

    Attributes:
    name: Name of the repository
    media_type: Media type of the data: application/json, application/csv, etc.
    modality: Modality of the data: image, text, etc.
    """

    name: str
    media_type: MediaTypes
    modality: Modality


class DataDataset(BaseDataModel):
    """Dataset model.

    Attributes:
    repository_id: Repository ID that identifies the repository(group of datasets)
    dataset_id: Dataset ID that identifies the dataset
    name: Name of the dataset
    labels: List of labels of the dataset
    total_datapoints: Total number of units in the dataset
    metadata: Metadata of the dataset
    created_at: Datetime when the dataset was created
    updated_at: Datetime when the dataset was updated
    """

    repository_id: str
    dataset_id: str
    name: Optional[str] = None
    labels: Optional[list[str]] = None
    total_datapoints: int
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class DatasetCreate(BaseDataModel):
    """Dataset creation model.

    Attributes:
    source_data: Source data of the dataset in bytes(file like object)
    name: Name of the dataset
    labels: List of labels of the dataset
    total_datapoints: Total number of units in the dataset
    metadata: Metadata of the dataset
    """

    model_config = ConfigDict(alias_generator=to_snake, arbitrary_types_allowed=True)

    source_data: io.BufferedReader | bytes
    name: Optional[str] = None
    labels: list[str]
    total_datapoints: int
    metadata: Optional[dict[str, Any]] = None
