import http
import os
from http import HTTPStatus
from typing import Annotated, Sequence

from aleph_alpha_client import Client
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.datastructures import URL

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import LuminousControlModel, NoOpTracer, Task
from intelligence_layer.use_cases import (
    ClassifyInput,
    PromptBasedClassify,
    SingleLabelClassifyOutput,
)

# Minimal FastAPI app ##########################################################

app = FastAPI()


@app.get("/")
def root() -> Response:
    return Response(content="Classification Service", status_code=HTTPStatus.OK)


# Authentication ###############################################################


class AuthService:
    def is_valid_token(self, token: str, permissions: Sequence[str], url: URL) -> bool:
        # Add your authentication logic here
        print(f"Checking permission for route: {url.path}")
        return True


class PermissionChecker:
    def __init__(self, permissions: Sequence[str] = []):
        self.permissions = permissions

    def __call__(
        self,
        request: Request,
        auth_service: Annotated[AuthService, Depends(AuthService)],
    ) -> None:
        token = request.headers.get("Authorization") or ""
        try:
            if not auth_service.is_valid_token(token, self.permissions, request.url):
                raise HTTPException(HTTPStatus.UNAUTHORIZED)
        except RuntimeError:
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR)


permission_checker_for_user = PermissionChecker(["User"])


# Intelligence Layer Task ######################################################

PROMPT = """Identify the department that would be responsible for handling the given request.
Reply with only the department name."""

load_dotenv()


def client() -> Client:
    return Client(
        token=os.environ["AA_TOKEN"],
        host=os.getenv("AA_CLIENT_BASE_URL", "https://api.aleph-alpha.com"),
    )


def default_model(
    app_client: Annotated[AlephAlphaClientProtocol, Depends(client)],
) -> LuminousControlModel:
    return LuminousControlModel("luminous-supreme-control", client=app_client)


def classification_task(
    model: Annotated[LuminousControlModel, Depends(default_model)],
) -> PromptBasedClassify:
    return PromptBasedClassify(instruction=PROMPT, model=model)


@app.post(
    "/classify",
    # dependencies=[Depends(PermissionChecker(["User"]))],
    status_code=http.HTTPStatus.OK,
)
def classification_task_route(
    input: ClassifyInput,
    task: Annotated[
        Task[ClassifyInput, SingleLabelClassifyOutput], Depends(classification_task)
    ],
) -> SingleLabelClassifyOutput:
    return task.run(input, NoOpTracer())
