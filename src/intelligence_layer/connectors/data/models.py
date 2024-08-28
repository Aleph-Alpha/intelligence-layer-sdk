import io
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel, to_snake


class BaseDataModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, arbitrary_types_allowed=True)


class DataRepository(BaseDataModel):
    """Data Repository model.

    Attributes:
    repository_id: Repository ID that identifies the repository(group of datasets)
    name: Name of the repository
    mutable: Indicates if the datasets in the repository are mutable or not
    mediaType: Media type of the data: application/json, application/csv, etc.
    modality: Modality of the data: image, text, etc.
    createdAt: Datetime when the repository was created
    updatedAt: Datetime when the repository was updated
    """

    repository_id: str
    name: str
    mutable: bool
    media_type: str
    modality: str
    created_at: datetime
    updated_at: datetime


class DataRepositoryCreate(BaseDataModel):
    """Data Repository creation model.

    Attributes:
    name: Name of the repository
    mediaType: Media type of the data: application/json, application/csv, etc.
    modality: Modality of the data: image, text, etc.
    """

    name: str
    media_type: str
    modality: str


class DataDataset(BaseDataModel):
    """Dataset model.

    Attributes:
    dataset_id: Dataset ID that identifies the dataset
    labels: List of labels of the dataset
    repository_id: Repository ID that identifies the repository(group of datasets)
    mutable: Indicates if the dataset is mutable or not
    total_datapoints: Total number of units in the dataset
    createdAt: Datetime when the dataset was created
    updatedAt: Datetime when the dataset was updated
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
    """

    model_config = ConfigDict(alias_generator=to_snake, arbitrary_types_allowed=True)

    source_data: io.BufferedReader | bytes
    name: Optional[str] = None
    labels: list[str]
    total_datapoints: int
    metadata: Optional[dict[str, Any]] = None
