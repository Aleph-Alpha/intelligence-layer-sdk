from os import getenv
from typing import TypeVar

import requests
from pydantic import BaseModel

from intelligence_layer.core import Task, TaskSpan

Input = TypeVar("Input", bound=BaseModel)
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output", bound=BaseModel)
"""Interface of the output returned by the task."""


class KernelTask(Task[Input, Output]):
    """A Task that can call a Skill within the Kernel.

    Note: this will not support full tracing in the Intelligence Layer,
    but it will allow passing a Kernel Skill as a subtask to a larger
    workflow, or allow for passing it to the Evaluation tooling.

    Args:
        skill: The name of the skill deployed in Pharia Kernel that should be called.
        input_model: The type for the Pydantic model that should be used for serializing the input.
        output_model: The type for the Pydantic model that should be used for deserializing the output.
        host: The URL to use for accessing Pharia Kernel. Defaults to the env variable `PHARIA_KERNEL_URL` if not provided.
        token: The auth token to use for accessing Pharia Kernel. Defaults to the env variable `AA_TOKEN` if not provided.
    """

    def __init__(
        self,
        skill: str,
        input_model: type[Input],
        output_model: type[Output],
        host: str | None = None,
        token: str | None = None,
    ):
        if host is None:
            host = getenv("PHARIA_KERNEL_URL")
            assert host, "Define PHARIA_KERNEL_URL with a valid url pointing towards your Pharia Kernel API."
        if token is None:
            token = getenv("AA_TOKEN")
            assert token, "Define environment variable AA_TOKEN with a valid token for the Aleph Alpha API"

        self.skill = skill
        self.input_model = input_model
        self.output_model = output_model
        self.host = host
        self.session = requests.Session()
        self.session.headers = {"Authorization": f"Bearer {token}"}

    def __del__(self):
        if self.session:
            self.session.close()

    def do_run(self, input: Input, task_span: TaskSpan) -> Output:
        response = self.session.post(
            f"{self.host}/v1/skills/{self.skill}/run",
            json=input.model_dump(),
        )

        if response.status_code != 200:
            raise Exception(f"{response.status_code}: {response.text}")

        return self.output_model(**response.json())
