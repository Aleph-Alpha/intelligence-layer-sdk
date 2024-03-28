# Tutorial: Extending a FastAPI App with the Aleph-Alpha Intelligence Layer

In this tutorial, a basic [FastAPI](https://fastapi.tiangolo.com) app is extended with a new route at which a summary for a given text can be retrieved, using the _Aleph-Alpha Intelligence Layer_, and it's _Luminous_ control models.

The full source code for this example app can be found at the end of this tutorial and in [src/examples/fastapi_example.py](./fastapi_example.py).

## Basic FastAPI App

The foundation for this tutorial is a minimal [FastAPI](https://fastapi.tiangolo.com) application with a root endpoint:

```python
from fastapi import FastAPI, Response
from http import HTTPStatus


app = FastAPI()

@app.get("/")
def root() -> Response:
    return Response(content="Hello World", status_code=HTTPStatus.OK)
```

This application can be started from the command line with the [Hypercorn](https://github.com/pgjones/hypercorn/) server as follows:

```bash
hypercorn fastapi_example:app --bind localhost:8000
```

If the start-up was successful, you should see a message similar to
```cmd
[2024-03-07 14:00:55 +0100] [6468] [INFO] Running on http://<your ip>:8000 (CTRL + C to quit)
```

Now that the server is running, we can perform a `GET` request via `cURL`:
```bash
curl -X GET http://localhost:8000
```
You should get
```
Hello World
```

After successfully starting the basic FastAPI app, the next step is to add a route that makes use of the Intelligence Layer.

## Adding the Intelligence Layer to the application

The building blocks of the Intelligence Layer for applications are `Tasks`. In general, a task implements the `Task`
interface and defines an `Input` and an `Output`. Multiple tasks can be chained to create more complex applications.
Here, we will make use of the pre-built task `SteerableSingleChunkSummarize`. This task defines `SingleChunkSummarizeInput`
as it's input, and `SummarizeOutput` as it's output.
Like many other tasks, the `SteerableSingleChunkSummarize` task makes use of a `ControlModel`. The
`ControlModel` itself needs access to the Aleph-Alpha backend via a `AlephAlphaClientProtocol` client.
In short, the hierarchy is as follows:

![task_dependencies.drawio.svg](task_dependencies.drawio.svg)

We make use of the built-in [Dependency Injection](https://fastapi.tiangolo.com/reference/dependencies/) of FastAPI to
resolve this hierarchy automatically. In this framework, the defaults for the parameters are dynamically created with
the `Depends(func)` annotation, where `func` is a function that returns the default value.

So, first, we define our client-generating function. For that, we provide the host URL and a valid Aleph-Alpha token,
which are stored in an `.env`-file.

```python
import os
from aleph_alpha_client import Client
from dotenv import load_dotenv

load_dotenv()

def client() -> Client:
    return Client(
        token=os.environ["AA_TOKEN"],
        host=os.getenv("AA_CLIENT_BASE_URL", "https://api.aleph-alpha.com")
    )
```

Next, we create a `ControlModel`. In this case, we make use of the `LuminousControlModel`, which takes
an `AlephAlphaClientProtocol` that we let default to the previously defined `client`.

```python
from typing import Annotated
from fastapi import Depends
from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import LuminousControlModel

def default_model(app_client: Annotated[AlephAlphaClientProtocol, Depends(client)]):
    return LuminousControlModel(client=app_client)
```


Finally, we create the actual `Task`. For our example, we choose the `SteerableSingleChunkSummarize`.
The `Input` of this task is a `SingleChunkSummarizeInput`, consisting of the text to summarize as the `chunk` field,
and the desired `Language` as the `language` field.
The `Output` of this task is a `SummarizeOutput` and contains the `summary` as text,
and number of generated tokens for the `summary` as the `generated_tokens` field.

```python
from intelligence_layer.use_cases import SteerableSingleChunkSummarize
from intelligence_layer.core import LuminousControlModel

def summary_task(
    model: Annotated[LuminousControlModel, Depends(default_model)],
) -> SteerableSingleChunkSummarize:
    return SteerableSingleChunkSummarize(model)
```
We can then provide a `POST` endpoint on `/summary` to run the task.
The default for `task` will be set by `summary_task`.

```python
from intelligence_layer.use_cases import (
    SingleChunkSummarizeInput,
    SummarizeOutput,
)
from intelligence_layer.core import NoOpTracer

@app.post("/summary")
def summary_task_route(
        input: SingleChunkSummarizeInput,
        task: Annotated[
            Task[SingleChunkSummarizeInput, SummarizeOutput],
            Depends(summary_task)
        ],
) -> SummarizeOutput:
    return task.run(input, NoOpTracer())
```

This concludes the addition of an Intelligence-Layer task to the FastAPI app. After restarting the server, we can call
our newly created  endpoint via a command such as the following:
```bash

curl -X POST http://localhost:8000/summary -H "Content-Type: application/json" -d '{"chunk": "<your text to summarize here>", "language": {"iso_639_1": "en"}}'
```

## Add Authorization to the Routes

Typically, authorization is needed to control access to endpoints.
Here, we will give a minimal example of how a per-route authorization system could be implemented in the minimal example app.

The authorization system makes use of two parts: An `AuthService` that checks whether the user is allowed to access a
given site, and a `PermissionsChecker` that is called on each route access and in turn calls the `AuthService`.

For this minimal example, the `AuthService` is simply a stub. You will want to implement a concrete authorization service
tailored to your needs.

```python
from typing import Sequence
from fastapi.datastructures import URL

class AuthService:
    def is_valid_token(
        self,
        token: str,
        permissions: Sequence[str],
        url: URL
    ):
        # Add your authentication logic here
        print(f"Checking permission for route: {url.path}")
        return True
```

With this `PermissionsChecker`, `permissions` can be passed in to define which roles, e.g. "user" or "admin",
are allowed to access which endpoints. The `PermissionsChecker` implements the `__call__` function, so that it can be
used as a function in the `dependencies` argument of each route via `Depends`. For more details see the extended
definition of the `summary_task_route` further below.

```python
from fastapi import HTTPException, Request

class PermissionChecker:
    def __init__(self, permissions: Sequence[str] = []):
        self.permissions = permissions

    def __call__(
        self,
        request: Request,
        auth_service:Annotated[AuthService, Depends(AuthService)],
    ) -> None:
        token = request.headers.get("Authorization")
        try:
            if not auth_service.is_valid_token(token, self.permissions, request.url):
                raise HTTPException(HTTPStatus.UNAUTHORIZED)
        except RuntimeError:
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR)
```

A specific `PermissionChecker` with `"User"` permissions is created which will be called for the `/summary` route to check, whether a "User" is allowed to access it.

```python
permission_checker_for_user = PermissionChecker(["User"])
```

The permission checker can be added to any route via the `dependencies` argument in the decorator. Here, we add it to the `summary_task_route`:

```python
@app.post("/summary", dependencies=[Depends(permission_checker_for_user)])
def summary_task_route(
    input: SingleChunkSummarizeInput,
    task: Annotated[
        Task[SingleChunkSummarizeInput, SummarizeOutput],
        Depends(summary_task)
    ],
) -> SummarizeOutput:
    return task.run(input, NoOpTracer())

```


## Complete Source

```python
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
    SingleChunkSummarizeInput,
    SteerableSingleChunkSummarize,
    SummarizeOutput,
)

# Minimal FastAPI app ##########################################################

app = FastAPI()


@app.get("/")
def root() -> Response:
    return Response(content="Hello World", status_code=HTTPStatus.OK)


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

load_dotenv()


def client() -> Client:
    return Client(
        token=os.environ["AA_TOKEN"],
        host=os.getenv("AA_CLIENT_BASE_URL", "https://api.aleph-alpha.com"),
    )


def default_model(
        app_client: Annotated[AlephAlphaClientProtocol, Depends(client)]
) -> LuminousControlModel:
    return LuminousControlModel(client=app_client)


def summary_task(
        model: Annotated[LuminousControlModel, Depends(default_model)],
) -> SteerableSingleChunkSummarize:
    return SteerableSingleChunkSummarize(model)


@app.post(
    "/summary",
    dependencies=[Depends(PermissionChecker(["User"]))],
    status_code=http.HTTPStatus.OK,
)
def summary_task_route(
        input: SingleChunkSummarizeInput,
        task: Annotated[
            Task[SingleChunkSummarizeInput, SummarizeOutput], Depends(summary_task)
        ],
) -> SummarizeOutput:
    return task.run(input, NoOpTracer())

```
