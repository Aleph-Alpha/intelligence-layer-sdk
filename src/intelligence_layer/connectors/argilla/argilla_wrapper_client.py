import itertools
import logging
import os
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Optional,
)

import argilla as rg  # type: ignore

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
            # this logger is set on info for some reason
            logging.getLogger("argilla.client.feedback.dataset.local.mixins").setLevel(
                logging.WARNING
            )

        self.client = rg.Argilla(
            api_url=api_url if api_url is not None else os.getenv("ARGILLA_API_URL"),
            api_key=api_key if api_key is not None else os.getenv("ARGILLA_API_KEY"),
        )

    def create_dataset(
        self,
        workspace_name: str,
        dataset_name: str,
        fields: Sequence[rg.TextField],
        questions: Sequence[rg.QuestionType],
    ) -> str:
        """Creates and publishes a new feedback dataset in Argilla.

        Raises an error if the name exists already.

        Args:
            workspace_name: the name of the workspace the feedback dataset should be created in.
                The user executing this request must have corresponding permissions for this workspace.
            dataset_name: the name of the feedback-dataset to be created.
            fields: all fields of this dataset.
            questions: all questions for this dataset.

        Returns:
            The id of the created dataset.
        """
        settings = rg.Settings(
            fields=list(fields),
            questions=list(questions),
            allow_extra_metadata=True,
        )

        workspace = self.ensure_workspace_exists(workspace_name)

        dataset = rg.Dataset(
            name=dataset_name,
            settings=settings,
            workspace=workspace,
            client=self.client,
        )
        dataset.create()
        return str(dataset.id)

    def ensure_dataset_exists(
        self,
        workspace_name: str,
        dataset_name: str,
        fields: Sequence[rg.TextField],
        questions: Sequence[rg.QuestionType],
    ) -> str:
        """Retrieves an existing dataset or creates and publishes a new feedback dataset in Argilla.

        Args:
            workspace_name: the name of the workspace the feedback dataset should be created in.
                The user executing this request must have corresponding permissions for this workspace.
            dataset_name: the name of the feedback-dataset to be created.
            fields: all fields of this dataset.
            questions: all questions for this dataset.

        Returns:
            The id of the dataset to be retrieved .
        """
        dataset = self.client.datasets(name=dataset_name, workspace=workspace_name)
        return (
            str(dataset.id)
            if dataset
            else self.create_dataset(workspace_name, dataset_name, fields, questions)
        )

    def add_record(self, dataset_id: str, record: RecordData) -> None:
        self.add_records(dataset_id=dataset_id, records=[record])

    def add_records(self, dataset_id: str, records: Sequence[RecordData]) -> None:
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)
        if remote_dataset is None:
            raise ValueError
        argilla_records = [
            rg.Record(
                fields=dict(record.content),
                metadata={
                    **record.metadata,
                    "example_id": record.example_id,
                },
            )
            for record in records
        ]
        remote_dataset.records.log(records=argilla_records)

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaEvaluation]:
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)

        status_filter = rg.Filter([("response.status", "==", "submitted")])
        query = rg.Query(filter=status_filter)

        for record in remote_dataset.records(query=query):
            metadata = record.metadata
            example_id = metadata.pop("example_id")
            yield ArgillaEvaluation(
                example_id=example_id,
                record_id="ignored",
                responses={
                    response.question_name: response.value
                    for response in record.responses
                    if response is not None
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

        metadata_config = remote_dataset.settings.metadata[name]

        if n_splits == 1:
            if metadata_config is None:
                return
            metadata_config.delete()
            self._delete_metadata_from_records(remote_dataset, name)
            return

        if metadata_config:
            metadata_config.delete()
            self._delete_metadata_from_records(remote_dataset, name)

        self._update_record_metadata(n_splits, remote_dataset, name)

    def _update_record_metadata(
        self, n_splits: int, remote_dataset: rg.Dataset, metadata_name: str
    ) -> None:
        config = rg.IntegerMetadataProperty(
            name=metadata_name, visible_for_annotators=True, min=1, max=n_splits
        )
        remote_dataset.settings.metadata = [config]
        remote_dataset.update()
        updated_records = []
        for record, split in zip(
            remote_dataset.records, itertools.cycle(range(1, n_splits + 1))
        ):
            record.metadata[metadata_name] = split
            updated_records.append(record)

        remote_dataset.records.log(updated_records)

    def _delete_metadata_from_records(
        self, remote_dataset: rg.Dataset, metadata_name: str
    ) -> None:
        modified_records = []
        for record in remote_dataset.records:
            del record.metadata[metadata_name]
            modified_records.append(record)
        remote_dataset.records.log(modified_records)

    def ensure_workspace_exists(self, workspace_name: str) -> str:
        """Retrieves the name of an argilla workspace with specified name or creates a new workspace if necessary.

        Args:
            workspace_name: the name of the workspace to be retrieved or created.

        Returns:
            The name of an argilla workspace with the given `workspace_name`.
        """
        workspace = self.client.workspaces(name=workspace_name)
        if workspace:
            return workspace_name

        workspace = rg.Workspace(name=workspace_name, client=self.client)
        workspace.create()
        if not workspace:
            raise ValueError(
                f"Workspace with name {workspace_name} could not be created."
            )
        return str(workspace.name)

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

    def _create_evaluation(
        self, dataset_id: str, record_id: str, data: dict[str, Any]
    ) -> None:
        dataset = self._dataset_from_id(dataset_id=dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset with id {dataset_id} does not exist.")
        records = dataset.records

        user_id = self.client.me.id
        if user_id is None:
            raise ValueError("user_id is not a UUID")

        # argilla currently does not allow to retrieve a record by id
        # This could be optimized (in a scenario for creating multiple evaluations) by passing a dict of record_ids to the function
        # and update all the records for the given record id list.
        for record in records:
            if record.id == record_id:
                for question_name, response_value in data.items():
                    response = rg.Response(
                        question_name=question_name,
                        value=response_value,
                        status="submitted",
                        user_id=user_id,
                    )
                    record.responses.add(response)
                dataset.records.log([record])
                return

    def _delete_dataset(self, dataset_id: str) -> None:
        remote_dataset = self._dataset_from_id(dataset_id=dataset_id)
        remote_dataset.delete()

    def _delete_workspace(self, workspace_name: str) -> None:
        workspace = self.client.workspaces(name=workspace_name)
        if workspace is None:
            raise ValueError("workspace does not exist.")
        for dataset in workspace.datasets:
            dataset.delete()
        workspace.delete()

    def _dataset_from_id(self, dataset_id: str) -> rg.Dataset:
        dataset = self.client.datasets(id=dataset_id)
        if not dataset:
            raise ValueError("Dataset is not existent")
        # Fetch Settings from Dataset in order to pass questions, necessary since Argilla V2
        dataset.settings.get()
        return dataset
