import itertools
import os
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import (
    Any,
)

import argilla as rg

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaRatingEvaluation,
    Field,
    Question,
    Record,
    RecordData,
)


class VanillaArgillaClient(ArgillaClient):
    def __init__(self) -> None:
        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY"),
        )

    def create_dataset(
        self,
        workspace_id: str,
        dataset_name: str,
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
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
        fields: Sequence[Field],
        questions: Sequence[Question],
    ) -> str:
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
        remote_dataset = rg.FeedbackDataset.from_argilla(id=dataset_id)
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

    def evaluations(self, dataset_id: str) -> Iterable[ArgillaRatingEvaluation]:
        remote_dataset = rg.FeedbackDataset.from_argilla(id=dataset_id)
        filtered_dataset = remote_dataset.filter_by(response_status="submitted")

        for record in filtered_dataset.records:
            submitted_response = next((response for response in record.responses), None)
            if submitted_response is not None:
                metadata = record.metadata
                example_id = metadata.pop("example_id")
                yield ArgillaRatingEvaluation(
                    example_id=example_id,
                    record_id="ignored",
                    responses={
                        k: v.value for k, v in submitted_response.values.items()
                    },
                    metadata=metadata,
                )

    def split_dataset(self, dataset_id: str, n_splits: int) -> None:
        remote_dataset = rg.FeedbackDataset.from_argilla(id=dataset_id)
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
        """Retrieves the id of an argilla workspace with specified name or creates a new workspace if necessary.

        Args:
            workspace_name: the name of the workspace to be retrieved or created.

        Returns:
            The id of an argilla workspace with the given `workspace_name`.
        """
        try:
            workspace = rg.Workspace.from_name(workspace_name)
            return str(workspace.name)
        except ValueError:
            return str(rg.Workspace.create(name=workspace_name).name)

    def records(self, dataset_id: str) -> Iterable[Record]:
        remote_dataset = rg.FeedbackDataset.from_argilla(id=dataset_id)
        return (
            Record(
                id=str(record.id),
                example_id=record.metadata["example_id"],
                content=record.fields,
                metadata=record.metadata,
            )
            for record in remote_dataset.records
        )

    def create_evaluation(
        self, dataset_id: str, example_id: str, data: dict[str, Any]
    ) -> None:
        # TODO this actually does not work, the patch request appears to simply not work correctly
        remote_dataset = rg.FeedbackDataset.from_argilla(id=dataset_id)
        filtered_dataset = remote_dataset.filter_by(
            metadata_filters=rg.TermsMetadataFilter(
                name="example_id", values=[example_id]
            )
        )
        modified_records = []
        for record in filtered_dataset.records:
            record.responses = [
                {
                    "values": {
                        question_name: {"value": response_value}
                        for question_name, response_value in data.items()
                    },
                    "status": "submitted",
                    "inserted_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
            ]
            modified_records.append(record)
        remote_dataset.update_records(modified_records, show_progress=False)
