import gzip
import json
import os
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar
from urllib.parse import urljoin
from uuid import UUID, uuid4

import requests
from pydantic import BaseModel, Field, RootModel, field_validator
from requests.exceptions import ConnectionError, MissingSchema

from intelligence_layer.connectors import JsonSerializable
from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core.tracer.tracer import (  # Import to be fixed with PHS-731
    ExportedSpan,
    ExportedSpanList,
    PydanticSerializable,
    Tracer,
)

Input = TypeVar("Input", bound=PydanticSerializable)
ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Output = TypeVar("Output", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=BaseModel, covariant=True)


class StudioProject(BaseModel):
    name: str
    description: Optional[str]


class StudioExample(BaseModel, Generic[Input, ExpectedOutput]):
    """Represents an instance of :class:`Example`as sent to Studio.

    Attributes:
        input: Input for the :class:`Task`. Has to be same type as the input for the task used.
        expected_output: The expected output from a given example run.
            This will be used by the evaluator to compare the received output with.
        id: Identifier for the example, defaults to uuid.
        metadata: Optional dictionary of custom key-value pairs.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
    """

    input: Input
    expected_output: ExpectedOutput
    id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: Optional[SerializableDict] = None


class StudioDataset(BaseModel):
    """Represents a :class:`Dataset` linked to multiple examples as sent to Studio.

    Attributes:
        id: Dataset ID.
        name: A short name of the dataset.
        label: Labels for filtering datasets. Defaults to empty list.
        metadata: Additional information about the dataset. Defaults to empty dict.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    labels: set[str] = set()
    metadata: SerializableDict = dict()


class EvaluationLogicIdentifier(BaseModel):
    logic: str  # code from the evaluation logic as a string
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    expected_output_schema: dict[str, Any]
    evaluation_schema: dict[str, Any]


class AggregationLogicIdentifier(BaseModel):
    logic: str  # code from the aggregation logic as a string
    evaluation_schema: dict[str, Any]
    aggregation_schema: dict[str, Any]


class PostBenchmarkRequest(BaseModel):
    dataset_id: str
    name: str
    description: Optional[str]
    benchmark_metadata: Optional[dict[str, Any]]
    evaluation_logic: EvaluationLogicIdentifier
    aggregation_logic: AggregationLogicIdentifier


class GetBenchmarkResponse(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    name: str
    description: str | None
    benchmark_metadata: dict[str, Any] | None
    evaluation_logic: EvaluationLogicIdentifier
    aggregation_logic: AggregationLogicIdentifier
    created_at: datetime
    updated_at: datetime | None
    last_executed_at: datetime | None
    created_by: str | None
    updated_by: str | None

    @field_validator("project_id", mode="before")
    def transform_id_to_str(cls, value) -> str:
        if type(value) is int or type(value) is UUID:
            return str(value)
        return value


class PostBenchmarkExecution(BaseModel):
    name: str
    description: Optional[str]
    labels: Optional[set[str]]
    metadata: Optional[dict[str, Any]]
    start: datetime
    end: datetime
    # Run Overview
    run_start: datetime
    run_end: datetime
    run_successful_count: int
    run_failed_count: int
    run_success_avg_latency: float
    run_success_avg_token_count: float
    # Eval Overview
    eval_start: datetime
    eval_end: datetime
    eval_successful_count: int
    eval_failed_count: int
    # Aggregation Overview
    aggregation_start: datetime
    aggregation_end: datetime
    statistics: JsonSerializable


class GetDatasetExamplesResponse(BaseModel, Generic[Input, ExpectedOutput]):
    total: int
    page: int
    size: int
    num_pages: int
    items: Sequence[StudioExample[Input, ExpectedOutput]]


class BenchmarkLineage(BaseModel, Generic[Input, ExpectedOutput, Output, Evaluation]):
    trace_id: str
    input: Input
    expected_output: ExpectedOutput
    output: Output
    example_metadata: Optional[dict[str, Any]] = None
    evaluation: Any
    run_latency: int
    run_tokens: int


class PostBenchmarkLineagesRequest(RootModel[Sequence[BenchmarkLineage]]):
    pass


class PostBenchmarkLineagesResponse(RootModel[Sequence[str]]):
    pass


class GetBenchmarkLineageResponse(BaseModel):
    id: str
    trace_id: str
    benchmark_execution_id: str
    input: JsonSerializable
    expected_output: JsonSerializable
    example_metadata: Optional[dict[str, JsonSerializable]] = None
    output: JsonSerializable
    evaluation: JsonSerializable
    run_latency: int
    run_tokens: int


class StudioClient:
    """Client for communicating with Studio.

    Attributes:
      project_id: The unique identifier of the project currently in use.
      url: The url of your current Studio instance.
    """

    @staticmethod
    def get_headers(auth_token: Optional[str] = None) -> dict[str, str]:
        _token = auth_token if auth_token is not None else os.getenv("AA_TOKEN")
        if _token is None:
            raise ValueError(
                "'AA_TOKEN' is not set and auth_token is not given as a parameter. Please provide one or the other."
            )
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {_token}",
        }

    @staticmethod
    def get_url(studio_url: Optional[str] = None) -> str:
        temp_url = studio_url if studio_url is not None else os.getenv("STUDIO_URL")
        if temp_url is None:
            raise ValueError(
                "'STUDIO_URL' is not set and url is not given as a parameter. Please provide one or the other."
            )
        return temp_url

    def __init__(
        self,
        project: str,
        studio_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        create_project: bool = False,
    ) -> None:
        """Initializes the client.

        Runs a health check to check for a valid url of the Studio connection.
        It does not check for a valid authentication token, which happens later.

        Args:
            project: The human readable identifier provided by the user.
            studio_url: The url of your current Studio instance.
            auth_token: The authorization bearer token of the user. This corresponds to the user's Aleph Alpha token.
            create_project: If True, the client will try to create the project if it does not exist. Defaults to False.
        """
        self._headers = StudioClient.get_headers(auth_token)
        self.url = StudioClient.get_url(studio_url)
        self._check_connection()
        self._project_name = project
        self._project_id: str | None = self._get_project(self._project_name)

        if create_project and self._project_id is None:
            self._project_id = self.create_project(
                self._project_name, reuse_existing=True
            )

    def _check_connection(self) -> None:
        try:
            url = urljoin(self.url, "/health")
            response = requests.get(
                url,
                headers=self._headers,
            )
            response.raise_for_status()
        except MissingSchema:
            raise ValueError(
                "The given url of the studio client is invalid. Make sure to include http:// in your url."
            ) from None
        except ConnectionError:
            raise ValueError(
                "The given url of the studio client does not point to a server."
            ) from None
        except requests.HTTPError:
            raise ValueError(
                f"The given url of the studio client does not point to a healthy studio: {response.status_code}: {response.json()}"
            ) from None

    @property
    def project_id(self) -> str:
        if self._project_id is None:
            project_id = self._get_project(self._project_name)
            if project_id is None:
                raise ValueError(
                    f"Project {self._project_name} was not available. Consider creating it with `StudioClient.create_project`."
                )
            self._project_id = project_id
        return self._project_id

    def _get_project(self, project_name: str) -> str | None:
        url = urljoin(self.url, "/api/projects")
        response = requests.get(
            url,
            headers=self._headers,
        )
        response.raise_for_status()
        all_projects = response.json()

        matching_projects = [
            proj for proj in all_projects if proj["name"] == project_name
        ]

        if not matching_projects:
            return None
        if len(matching_projects) > 1:
            raise ValueError(
                f"Multiple projects with name '{project_name}' found. Please make sure project names available to you are unique. E.g. rename/delete the duplicated project"
            )

        project_of_interest = matching_projects[0]
        # Studio API service < v0.1.0 does not have a "project_id" field
        if "project_id" in project_of_interest:
            return str(project_of_interest["project_id"])
        return str(project_of_interest["id"])

    def create_project(
        self,
        project: str,
        description: Optional[str] = None,
        reuse_existing: bool = False,
    ) -> str:
        """Creates a project in Studio.

        Projects are uniquely identified by the user provided name.

        Args:
            project: User provided name of the project.
            description: Description explaining the usage of the project. Defaults to None.
            reuse_existing: Reuse project with specified name if already existing. Defaults to False.


        Returns:
            The ID of the newly created project.
        """
        if reuse_existing:
            fetched_project = self._get_project(project)
            if fetched_project is not None:
                return fetched_project
        url = urljoin(self.url, "/api/projects")
        data = StudioProject(name=project, description=description)
        response = requests.post(
            url,
            data=data.model_dump_json(),
            headers=self._headers,
        )
        response.raise_for_status()
        return str(response.json())

    def _delete_project(
        self,
        project_id: str,
    ) -> None:
        url = urljoin(self.url, f"/api/projects/{project_id}")
        response = requests.delete(
            url,
            headers=self._headers,
        )
        response.raise_for_status()

    def submit_trace(self, data: Sequence[ExportedSpan]) -> str:
        """Sends the provided spans to Studio as a singular trace.

        The method fails if the span list is empty, has already been created or if
        spans belong to multiple traces.

        Args:
            data: :class:`Spans` to create the trace from. Created by exporting from a :class:`Tracer`.

        Returns:
            The ID of the created trace.
        """
        if len(data) == 0:
            raise ValueError("Tried to upload an empty trace")
        return self._upload_trace(ExportedSpanList(data))

    def submit_from_tracer(self, tracer: Tracer) -> list[str]:
        """Sends all trace data from the Tracer to Studio.

        Args:
            tracer: :class:`Tracer` to extract data from.

        Returns:
            List of created trace IDs.
        """
        traces = defaultdict(list)
        for span in tracer.export_for_viewing():
            traces[span.context.trace_id].append(span)

        return [self.submit_trace(value) for value in traces.values()]

    def _upload_trace(self, trace: ExportedSpanList) -> str:
        url = urljoin(self.url, f"/api/projects/{self.project_id}/traces")
        response = requests.post(
            url,
            data=trace.model_dump_json(),
            headers=self._headers,
        )
        match response.status_code:
            case 409:
                raise ValueError(
                    f"Trace with id {trace.root[0].context.trace_id} already exists."
                )
            case 422:
                raise ValueError(
                    f"Uploading the trace failed with 422. Response: {response.json()}"
                )
            case _:
                response.raise_for_status()
        return str(response.json())

    def submit_dataset(
        self,
        dataset: StudioDataset,
        examples: Iterable[StudioExample[Input, ExpectedOutput]],
    ) -> str:
        """Submits a dataset to Studio.

        Args:
            dataset: :class:`Dataset` to be uploaded
            examples: :class:`Examples` of the :class:`Dataset`

        Returns:
            ID of the created dataset
        """
        url = urljoin(self.url, f"/api/projects/{self.project_id}/evaluation/datasets")
        source_data_list = [example.model_dump_json() for example in examples]

        source_data_file = "\n".join(source_data_list).encode()

        data = {
            "name": dataset.name,
            "labels": list(dataset.labels) if dataset.labels is not None else [],
            "total_datapoints": len(source_data_list),
        }

        if dataset.metadata:
            data["metadata"] = json.dumps(dataset.metadata)

        response = requests.post(
            url,
            files={"source_data": source_data_file},
            data=data,
            headers=self._headers,
        )

        self._raise_for_status(response)
        return str(response.json())

    def get_dataset_examples(
        self,
        dataset_id: str,
        input_type: type[Input],
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[StudioExample[Input, ExpectedOutput]]:
        buffer_size = 200
        page_size = 100
        page: int | None = 1
        buffer: deque[StudioExample[Input, ExpectedOutput]] = deque()

        while True:
            if len(buffer) < buffer_size // 2 and page is not None:
                page_url = urljoin(
                    self.url,
                    f"/api/projects/{self.project_id}/evaluation/datasets/{dataset_id}/datapoints?page={page}&size={page_size}",
                )

                response = requests.get(page_url, headers=self._headers)

                if response.status_code == 200:
                    examples = GetDatasetExamplesResponse(**response.json()).items
                    buffer.extend(examples)

                    if len(examples) < page_size:
                        page = None
                    else:
                        page += 1
                else:
                    raise Exception(
                        f"Failed to fetch items from {page_url}. Status code: {response.status_code}"
                    )

            if len(buffer) > 0:
                yield StudioExample[
                    input_type, expected_output_type  # type: ignore
                ].model_validate_json(json_data=buffer.popleft().model_dump_json())
            else:
                if page is None:
                    break

    def submit_benchmark(
        self,
        dataset_id: str,
        eval_logic: EvaluationLogicIdentifier,
        aggregation_logic: AggregationLogicIdentifier,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        benchmark = PostBenchmarkRequest(
            dataset_id=dataset_id,
            name=name,
            description=description,
            benchmark_metadata=metadata,
            evaluation_logic=eval_logic,
            aggregation_logic=aggregation_logic,
        )
        url = urljoin(
            self.url, f"/api/projects/{self.project_id}/evaluation/benchmarks"
        )
        response = requests.post(
            url,
            data=benchmark.model_dump_json(),
            headers=self._headers,
        )
        self._raise_for_status(response)
        return str(response.json())

    def get_benchmark(
        self,
        benchmark_id: str,
    ) -> GetBenchmarkResponse | None:
        url = urljoin(
            self.url,
            f"/api/projects/{self.project_id}/evaluation/benchmarks/{benchmark_id}",
        )
        response = requests.get(
            url,
            headers=self._headers,
        )
        self._raise_for_status(response)
        response_text = response.json()
        if response_text is None:
            return None
        return GetBenchmarkResponse.model_validate(response_text)

    def submit_benchmark_execution(
        self, benchmark_id: str, data: PostBenchmarkExecution
    ) -> str:
        url = urljoin(
            self.url,
            f"/api/projects/{self.project_id}/evaluation/benchmarks/{benchmark_id}/executions",
        )

        response = requests.post(
            url, headers=self._headers, data=data.model_dump_json()
        )

        self._raise_for_status(response)
        return str(response.json())

    def submit_benchmark_lineages(
        self,
        benchmark_lineages: Sequence[BenchmarkLineage],
        benchmark_id: str,
        execution_id: str,
        max_payload_size: int = 50
        * 1024
        * 1024,  # Maximum request size handled by Studio
    ) -> PostBenchmarkLineagesResponse:
        """Submit benchmark lineages in batches to avoid exceeding the maximum payload size.

        Args:
            benchmark_lineages: List of :class: `BenchmarkLineages` to submit.
            benchmark_id: ID of the benchmark.
            execution_id: ID of the execution.
            max_payload_size: Maximum size of the payload in bytes. Defaults to 50MB.

        Returns:
            Response containing the results of the submissions.
        """
        all_responses = []
        remaining_lineages = list(benchmark_lineages)
        lineage_sizes = [
            len(lineage.model_dump_json().encode("utf-8"))
            for lineage in benchmark_lineages
        ]

        while remaining_lineages:
            batch = []
            current_size = 0
            # Build batch while checking size
            for lineage, size in zip(remaining_lineages, lineage_sizes, strict=True):
                if current_size + size <= max_payload_size:
                    batch.append(lineage)
                    current_size += size
                else:
                    break

            if batch:
                # Send batch
                response = self._send_compressed_batch(
                    batch, benchmark_id, execution_id
                )
                all_responses.extend(response)

            else:  # Only reached if a lineage is too big for the request
                print("Lineage exceeds maximum of upload size", lineage)
                batch.append(lineage)
            remaining_lineages = remaining_lineages[len(batch) :]
            lineage_sizes = lineage_sizes[len(batch) :]

        return PostBenchmarkLineagesResponse(all_responses)

    def get_benchmark_lineage(
        self, benchmark_id: str, execution_id: str, lineage_id: str
    ) -> GetBenchmarkLineageResponse | None:
        url = urljoin(
            self.url,
            f"/api/projects/{self.project_id}/evaluation/benchmarks/{benchmark_id}/executions/{execution_id}/lineages/{lineage_id}",
        )
        response = requests.get(
            url,
            headers=self._headers,
        )
        self._raise_for_status(response)
        response_text = response.json()
        if response_text is None:
            return None
        return GetBenchmarkLineageResponse.model_validate(response_text)

    def _send_compressed_batch(
        self, batch: list[BenchmarkLineage], benchmark_id: str, execution_id: str
    ) -> list[str]:
        url = urljoin(
            self.url,
            f"/api/projects/{self.project_id}/evaluation/benchmarks/{benchmark_id}/executions/{execution_id}/lineages",
        )

        json_data = PostBenchmarkLineagesRequest(root=batch).model_dump_json()
        compressed_data = gzip.compress(json_data.encode("utf-8"))

        headers = {**self._headers, "Content-Encoding": "gzip"}

        response = requests.post(
            url,
            headers=headers,
            data=compressed_data,
        )

        self._raise_for_status(response)
        return response.json()

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print(
                f"The following error has been raised via execution {e.response.text}"
            )
            raise e
