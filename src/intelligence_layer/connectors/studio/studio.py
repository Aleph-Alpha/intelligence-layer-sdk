import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Optional
from urllib.parse import urljoin

import requests
from pydantic import BaseModel
from requests.exceptions import ConnectionError, MissingSchema

from intelligence_layer.core.tracer.tracer import ExportedSpan, ExportedSpanList, Tracer


class StudioProject(BaseModel):
    name: str
    description: Optional[str]


class StudioClient:
    """Client for communicating with PhariaStudio.

    Attributes:
      project_id: The unique identifier of the project currently in use.
      url: The url of your current PhariaStudio instance.
    """

    def __init__(
        self,
        project: str,
        studio_url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        """Initializes the client.

        Runs a health check to check for a valid url of the Studio connection.
        It does not check for a valid authentication token, which happens later.

        Args:
            project: The human readable identifier provided by the user.
            studio_url: The url of your current PhariaStudio instance.
            auth_token: The authorization bearer token of the user. This corresponds to the user's Aleph Alpha token.
        """
        self._token = auth_token if auth_token is not None else os.getenv("AA_TOKEN")
        if self._token is None:
            raise ValueError(
                "'AA_TOKEN' is not set and auth_token is not given as a parameter. Please provide one or the other."
            )
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }

        temp_url = studio_url if studio_url is not None else os.getenv("STUDIO_URL")
        if temp_url is None:
            raise ValueError(
                "'STUDIO_URL' is not set and url is not given as a parameter. Please provide one or the other."
            )
        self.url = temp_url

        self._check_connection()

        self._project_name = project
        self._project_id: int | None = None

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
        if len(data) == 0:
            raise ValueError("Tried to upload an empty trace")
        return self._upload_trace(ExportedSpanList(data))

    def submit_from_tracer(self, tracer: Tracer) -> list[str]:
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
