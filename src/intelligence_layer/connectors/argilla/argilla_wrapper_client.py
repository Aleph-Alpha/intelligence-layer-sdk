import itertools
import os
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Optional,
)

import argilla as rg  # type: ignore
from argilla.client.feedback.schemas.types import (  # type: ignore
    AllowedFieldTypes,
    AllowedQuestionTypes,
)

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Record,
    RecordData,
)


class ArgillaWrapperClient(ArgillaClient):
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        disable_warnings: bool = True,
    ) -> None:
        if disable_warnings:
            import warnings

            warnings.filterwarnings("ignore", module="argilla.*")
        rg.init(
            api_url=api_url if api_url is not None else os.getenv("ARGILLA_API_URL"),
            api_key=api_key if api_key is not None else os.getenv("ARGILLA_API_KEY"),
        )

    def create_dataset(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[AllowedFieldTypes],
        questions: Sequence[AllowedQuestionTypes],
    ) -> str:
        """Creates and publishes a new feedback dataset in Argilla.

        Raises an error if the name exists already.

        Args:
            workspace_id: the name of the workspace the feedback dataset should be created in.
                The user executing this request must have corresponding permissions for this workspace.
            dataset_name: the name of the feedback-dataset to be created.
            fields: all fields of this dataset.
            questions: all questions for this dataset.

        Returns:
            The id of the created dataset.
        """
        ...
        dataset = rg.FeedbackDataset(
            fields=fields, questions=questions, allow_extra_metadata=True
        )
        remote_datasets = dataset.push_to_argilla(
            name=dataset_name,
            workspace=rg.Workspace.from_name(workspace_id),
            show_progress=False,
        )
        return str(remote_datasets.id)

    def ensure_dataset_exists(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[AllowedFieldTypes],
        questions: Sequence[AllowedQuestionTypes],
    ) -> str:
        """Retrieves an existing dataset or creates and publishes a new feedback dataset in Argilla.

        Args:
            workspace_id: the name of the workspace the feedback dataset should be created in.
                The user executing this request must have corresponding permissions for this workspace.
            dataset_name: the name of the feedback-dataset to be created.
            fields: all fields of this dataset.
            questions: all questions for this dataset.

        Returns:
            The id of the dataset to be retrieved .
        """
        try:
            return str(
                rg.FeedbackDataset.from_argilla(
                    name=dataset_name, workspace=rg.Workspace.from_name(workspace_id)
                ).id
            )
        except ValueError:
            pass
        return self.create_dataset(workspace_id, dataset_name, fields, questions)

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        self.add_records(dataset_id=dataset_id, records=[record])

    def add_records(self, dataset_id: str, records: Sequence[RecordData]) -> None:
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)
        argilla_records = [
            rg.FeedbackRecord(
                fields=record.content,
                metadata={
                    **record.metadata,
                    "example_id": record.example_id,
                },
            )
            for record in records
        ]
        remote_dataset.add_records(argilla_records, show_progress=False)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)
        filtered_dataset = remote_dataset.filter_by(response_status="submitted")

        for record in filtered_dataset.records:
            submitted_response = next((response for response in record.responses), None)
            if submitted_response is not None:
                metadata = record.metadata
                example_id = metadata.pop("example_id")
                yield ArgillaEvaluation(
                    example_id=example_id,
                    record_id="ignored",
                    responses={
                        k: v.value for k, v in submitted_response.values.items()
                    },
                    metadata=metadata,
                )

    def split_dataset(self, dataset_id: str, n_splits: int) -> None:
        """Adds a new metadata property to the dataset and assigns a split to each record.

        Deletes the property if n_splits is equal to one.

        Args:
            dataset_id: the id of the dataset
            n_splits: the number of splits to create
        """
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)
        name = "split"
        metadata_config = remote_dataset.metadata_property_by_name(name)
        if metadata_config is None:
            if n_splits == 1:
                return
            config = rg.IntegerMetadataProperty(
                name=name, visible_for_annotators=True, min=1, max=n_splits
            )
            remote_dataset.add_metadata_property(config)
        else:
            if n_splits == 1:
                remote_dataset.delete_metadata_properties(name)
                modified_records = []
                for record in remote_dataset.records:
                    del record.metadata[name]
                    modified_records.append(record)
                remote_dataset.update_records(modified_records, show_progress=False)
                return
            else:
                metadata_config.max = n_splits
                remote_dataset.update_metadata_properties(metadata_config)

        modified_records = []
        for record, split in zip(
            remote_dataset.records, itertools.cycle(range(1, n_splits + 1))
        ):
            record.metadata[name] = split
            modified_records.append(record)
        remote_dataset.update_records(modified_records, show_progress=False)

    def ensure_workspace_exists(self, workspace_name: str) -> str:
        """Retrieves the name of an argilla workspace with specified name or creates a new workspace if necessary.

        Args:
            workspace_name: the name of the workspace to be retrieved or created.

        Returns:
            The name of an argilla workspace with the given `workspace_name`.
        """
        try:
            workspace = rg.Workspace.from_name(workspace_name)
            return str(workspace.name)
        except ValueError:
            return str(rg.Workspace.create(name=workspace_name).name)

    def records(self, dataset_id: str) -> Iterable[Record]:
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)
        return (
            Record(
                id=str(record.id),
                example_id=record.metadata["example_id"],
                content=record.fields,
                metadata=record.metadata,
            )
            for record in remote_dataset.records
        )

    def _create_evaluation(self, record_id: str, data: dict[str, Any]) -> None:
        api_url = os.environ["ARGILLA_API_URL"]
        if not api_url.endswith("/"):
            api_url = api_url + "/"
        rg.active_client().http_client.post(
            f"{api_url}api/v1/records/{record_id}/responses",
            json={
                "status": "submitted",
                "values": {
                    question_name: {"value": response_value}
                    for question_name, response_value in data.items()
                },
            },
        )

    def _delete_dataset(self, dataset_id: str) -> None:
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)
        remote_dataset.delete()

    def _delete_workspace(self, workspace_name: str) -> None:
        workspace = rg.Workspace.from_name(workspace_name)
        datasets = rg.list_datasets(workspace=workspace.name)
        for dataset in datasets:
            dataset.delete()
        workspace.delete()

    def _dataset_from_id(self, dataset_id: str) -> rg.FeedbackDataset:
        return rg.FeedbackDataset.from_argilla(id=dataset_id)
