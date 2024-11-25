import json
import os
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar
from urllib.parse import urljoin
from uuid import uuid4

import requests
from pydantic import BaseModel, Field
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
    project_id: int
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
    run_success_avg_latency: int
    run_success_avg_token_count: int
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
        self._project_id: int | None = None

        if create_project:
            project_id = self._get_project(self._project_name)
            if project_id is None:
                self.create_project(self._project_name)
            self._project_id = project_id

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
    def project_id(self) -> int:
        if self._project_id is None:
            project_id = self._get_project(self._project_name)
            if project_id is None:
                raise ValueError(
                    f"Project {self._project_name} was not available. Consider creating it with `StudioClient.create_project`."
                )
            self._project_id = project_id
        return self._project_id

    def _get_project(self, project: str) -> int | None:
        url = urljoin(self.url, "/api/projects")
        response = requests.get(
            url,
            headers=self._headers,
        )
        response.raise_for_status()
        all_projects = response.json()
        try:
            project_of_interest = next(
                proj for proj in all_projects if proj["name"] == project
            )
            return int(project_of_interest["id"])
        except StopIteration:
            return None

    def create_project(self, project: str, description: Optional[str] = None) -> int:
        """Creates a project in Studio.

        Projects are uniquely identified by the user provided name.

        Args:
            project: User provided name of the project.
            description: Description explaining the usage of the project. Defaults to None.

        Returns:
            The ID of the newly created project.
        """
        url = urljoin(self.url, "/api/projects")
        data = StudioProject(name=project, description=description)
        response = requests.post(
            url,
            data=data.model_dump_json(),
            headers=self._headers,
        )
        match response.status_code:
            case 409:
                raise ValueError("Project already exists")
            case _:
                response.raise_for_status()
        return int(response.text)

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

    def create_benchmark(
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

    def create_benchmark_execution(
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

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print(
                f"The following error has been raised via execution {e.response.text}"
            )
            raise e
