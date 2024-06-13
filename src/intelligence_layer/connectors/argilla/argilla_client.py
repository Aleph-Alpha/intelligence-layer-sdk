from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    Any,
    Union,
)

from pydantic import BaseModel, computed_field
from pydantic import Field as PydanticField


class Field(BaseModel):
    """Definition of an Argilla feedback-dataset field.

    Attributes:
        name: The name of the field. This is used to reference the field in json-documents
        title: The title of the field. This is displayed in the Argilla UI to users that perform the manual evaluations.

    """

    name: str
    title: str


class Question(BaseModel):
    """Definition of an evaluation-question for an Argilla feedback dataset.

    Attributes:
        name: The name of the question. This is used to reference the questions in json-documents
        title: The title of the field. This is displayed in the Argilla UI to users that perform the manual evaluations.
        description: A more verbose description of the question.
            This is displayed in the Argilla UI to users that perform the manual evaluations.
    """

    name: str
    title: str
    description: str

    @computed_field  # type: ignore[misc]
    @property
    def settings(self) -> Mapping[Any, Any]:
        raise NotImplementedError("")


class RatingQuestion(Question):
    """Definition of a rating evaluation-question for an Argilla feedback dataset.

    Attributes:
        options: All integer options to answer this question
    """

    options: Sequence[int]  # range: 1-10

    @computed_field  # type: ignore[misc]
    @property
    def settings(self) -> Mapping[str, Any]:
        return {
            "type": "rating",
            "options": [{"value": option} for option in self.options],
        }


class TextQuestion(Question):
    """Definition of a text evaluation-question for an Argilla feedback dataset.

    Attributes:
        use_markdown: Set this parameter to True if you want to use markdown
    """

    use_markdown: bool

    @computed_field  # type: ignore[misc]
    @property
    def settings(self) -> Mapping[str, Any]:
        return {"type": "text", "use_markdown": self.use_markdown}


class ArgillaRatingEvaluation(BaseModel):
    """The evaluation result for a single rating record in an Argilla feedback-dataset.

    Attributes:
        example_id: the id of the example that was evaluated.
        record_id: the id of the record that is evaluated.
        responses: Maps question-names (:attr:`Question.name` ) to response values.
        metadata: Metadata belonging to the evaluation, for example ids of completions.
    """

    example_id: str
    record_id: str
    responses: Mapping[str, Union[str, int, float, bool]]
    metadata: Mapping[str, str]


class RecordData(BaseModel):
    """Input-data for a Argilla evaluation record.

    This can be used to add a new record to an existing Argilla feedback-dataset.
    Once it is added it gets an Argilla provided id and can be retrieved as :class:`Record`

    Attributes:
        content: Maps field-names (:attr:`Field.name` ) to string values that can be displayed to the user.
        example_id: the id of the corresponding :class:`Example` from a :class:`Dataset`.
        metadata: Arbitrary metadata in form of key/value strings that can be attached to a record.
    """

    content: Mapping[str, str]
    example_id: str
    metadata: Mapping[str, str | int] = PydanticField(default_factory=dict)


class Record(RecordData):
    """Represents an Argilla record of an feedback-dataset.

    Just adds the id to a :class:`RecordData`

    Attributes:
        id: the Argilla generated id of the record.
    """

    id: str


class ArgillaClient(ABC):
    """Client interface for accessing an Argilla server.

    Argilla supports human in the loop evaluation. This class defines the API used by
    the intelligence layer to create feedback datasets or retrieve evaluation results.
    """

    @abstractmethod
    def create_dataset(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        """Creates and publishes a new feedback dataset in Argilla.

        Raises an error if the name exists already.

        Args:
            workspace_id: the id of the workspace the feedback-dataset should be created in.
                The user executing this request must have corresponding permissions for this workspace.
            dataset_name: the name of the feedback-dataset to be created.
            fields: all fields of this dataset.
            questions: all questions for this dataset.

        Returns:
            The id of the created dataset.
        """
        ...

    @abstractmethod
    def ensure_dataset_exists(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
        """Retrieves an existing dataset or creates and publishes a new feedback dataset in Argilla.

        Args:
            workspace_id: the id of the workspace the feedback-dataset should be created in.
                The user executing this request must have corresponding permissions for this workspace.
            dataset_name: the name of the feedback-dataset to be created.
            fields: all fields of this dataset.
            questions: all questions for this dataset.

        Returns:
            The id of the dataset to be retrieved .
        """
        ...

    @abstractmethod
    def add_record(self, dataset_id: str, record: RecordData) -> None:
        """Adds a new record to the given dataset.

        Args:
            dataset_id: id of the dataset the record is added to
            record: the actual record data (i.e. content for the dataset's fields)
        """
        ...

    def add_records(self, dataset_id: str, records: Sequence[RecordData]) -> None:
        """Adds new records to the given dataset.

        Args:
            dataset_id: id of the dataset the record is added to
            records: list containing the record data (i.e. content for the dataset's fields)
        """
        for record in records:
            return self.add_record(dataset_id, record)

    @abstractmethod
    def evaluations(self, dataset_id: str) -> Iterable[ArgillaRatingEvaluation]:
        """Returns all human-evaluated evaluations for the given dataset.

        Args:
            dataset_id: the id of the dataset.

        Returns:
            An `Iterable` over all human-evaluated evaluations for the given dataset.
        """
        ...

    @abstractmethod
    def split_dataset(self, dataset_id: str, n_splits: int) -> None:
        """Adds a new metadata property to the dataset and assigns a split to each record.

        Args:
            dataset_id: the id of the dataset
            n_splits: the number of splits to create
        """
        ...
