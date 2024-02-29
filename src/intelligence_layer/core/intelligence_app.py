from abc import ABC, abstractmethod
from functools import partial
from http import HTTPStatus
from typing import Annotated, Callable, Generic, TypeVar

from fastapi import Body, Depends, FastAPI, HTTPException, Response, status
from uvicorn import run

from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer.tracer import NoOpTracer, Tracer

T = TypeVar("T")


class AuthService(ABC, Generic[T]):
    @abstractmethod
    def get_permissions(
        self,
        required_permissions: frozenset[str],
        credentials: Annotated[T, None],
    ) -> bool:
        pass


class NoAuthService(AuthService[None]):
    def get_permissions(
        self,
        _: frozenset[str],
        __: Annotated[None, Depends(lambda: None)],
    ) -> bool:
        return True


class IntelligenceApp:
    """The Intelligence App is an easy way to turn your tasks into an application.

    This app is used to quickly setup a FastAPI server with any registered tasks that you would like.
    By registering your tasks you can expose them as endpoints ready for usage.

    Args:
        fast_api_app: This is the FastAPI app the IntelligenceApp relies on for routing.
    """

    def __init__(self, fast_api_app: FastAPI) -> None:
        self._fast_api_app = fast_api_app

    def register_task(
        self,
        task: Task[Input, Output] | Callable[..., Task[Input, Output]],
        input_type: type[Input],
        path: str,
    ) -> None:
        """Registers a task to your application.

        Registering a task will make it available as an endpoint.
        For technical reasons, your endpoint cannot have a `TaskSpan` as input.

        Args:
            task: The task you would like expose. This is either a direct instance of the task
                or a factory function that is used as a FastAPI dependency in the
                added FastAPI route.
            input_type: the type of the task's input parameter (i.e. first parameter of :meth:`Task.do_run`)
            path: The path your exposed endpoint will have.

        Example:
            >>> import os
            >>> from aleph_alpha_client import Client
            >>> from fastapi import FastAPI
            >>> from intelligence_layer.core import IntelligenceApp, CompleteInput, AlephAlphaModel
            >>> from intelligence_layer.connectors import LimitedConcurrencyClient

            >>> fast_api = FastAPI()
            >>> app = IntelligenceApp(fast_api)
            >>> aa_client = LimitedConcurrencyClient.from_env()
            >>> model = AlephAlphaModel("luminous-base", client=aa_client)
            >>> app.register_task(model.complete_task(), CompleteInput, "/complete")
        """
        # mypy does not like the dynamic input_type as type-parameter
        if isinstance(task, Task):

            @self._fast_api_app.post(path)
            def task_route(input: Annotated[input_type, Body()]) -> Output:  # type: ignore
                return _run_task(task, input, NoOpTracer())

        else:

            @self._fast_api_app.post(path)
            def task_route(
                input: Annotated[input_type, Body()],  # type: ignore
                task: Annotated[Task[Input, Output], Depends(task)],
            ) -> Output:
                return _run_task(task, input, NoOpTracer())

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """This starts the application.

        Args:
            host: The base url where the application will be served on.
            port: The port the application will listen to.
        """
        run(self._fast_api_app, host=host, port=port)


class AuthenticatedIntelligenceApp(IntelligenceApp, Generic[T]):
    def __init__(self, fast_api_app: FastAPI, auth_service: AuthService[T]) -> None:
        super().__init__(fast_api_app)
        self._auth_service = auth_service

    def register_task(
        self,
        task: Task[Input, Output] | Callable[..., Task[Input, Output]],
        input_type: type[Input],
        path: str,
        required_permissions: frozenset[str] = frozenset(),
    ) -> None:
        has_permission = Depends(
            partial(self._auth_service.get_permissions, required_permissions)
        )
        # mypy does not like the dynamic input_type as type-parameter
        if isinstance(task, Task):

            @self._fast_api_app.post(path)
            def task_route(
                input: Annotated[input_type, Body()],  # type: ignore
                allowed: Annotated[bool, has_permission],
            ) -> Output:
                return _run_task(task, input, NoOpTracer(), allowed)

        else:

            @self._fast_api_app.post(path)
            def task_route(
                input: Annotated[input_type, Body()],  # type: ignore
                task: Annotated[Task[Input, Output], Depends(task)],
                allowed: Annotated[bool, has_permission],
            ) -> Output:
                return _run_task(task, input, NoOpTracer(), allowed)


def _run_task(
    task: Task[Input, Output], input: Input, tracer: Tracer, allowed: bool = True
) -> Output:
    if allowed:
        output = task.run(input, tracer)
        return Response(status_code=HTTPStatus.NO_CONTENT) if output is None else output
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No permission rights",
    )
