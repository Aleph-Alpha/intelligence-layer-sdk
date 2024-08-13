import io
from datetime import datetime

from pydantic import BaseModel


class DataRepository(BaseModel):
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
    mediaType: str
    modality: str
    createdAt: datetime
    updatedAt: datetime


class DataRepositoryCreate(BaseModel):
    """Data Repository creation model.

    Attributes:
    name: Name of the repository
    mediaType: Media type of the data: application/json, application/csv, etc.
    modality: Modality of the data: image, text, etc.
    """

    name: str
    mediaType: str
    modality: str


class Dataset(BaseModel):
    """Dataset model.

    Attributes:
    dataset_id: Dataset ID that identifies the dataset
    labels: List of labels of the dataset
    repository_id: Repository ID that identifies the repository(group of datasets)
    mutable: Indicates if the dataset is mutable or not
    total_units: Total number of units in the dataset
    createdAt: Datetime when the dataset was created
    updatedAt: Datetime when the dataset was updated
    """

    repository_id: str
    dataset_id: str
    labels: list[str]
    total_units: int
    created_at: str
    updated_at: str


class DatasetCreate(BaseModel):
    """Dataset creation model.

    Attributes:
    source_data: Source data of the dataset in bytes(file like object)
    name: Name of the dataset
    labels: List of labels of the dataset
    """

    source_data: io.BufferedReader | bytes
    labels: list[str]
    total_units: int

    class Config:
        arbitrary_types_allowed = True
