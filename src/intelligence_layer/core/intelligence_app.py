from abc import ABC, abstractmethod
from functools import partial
from inspect import get_annotations
from typing import Annotated, Any, Generic, TypeVar

from fastapi import Body, Depends, FastAPI, HTTPException, status
from uvicorn import run

from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import NoOpTracer, TaskSpan


class RegisterTaskError(TypeError):
    """Error raised when incorrectly initialising a task for an IntelligenceApp."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


T = TypeVar("T")


class AuthService(ABC, Generic[T]):
    @abstractmethod
    def get_permissions(
        self,
        required_permissions: frozenset[str],
        credentials: Annotated[T, None],
    ) -> bool:
        ...


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

    DO_RUN_NEEDS_TYPE_HINTS_FOR_INPUT_AND_TASK_SPAN = (
        "The task `do_run` method needs a type at least for its input and task_span."
    )
    DO_RUN_MUST_HAVE_SINGLE_TASKSPAN_ARGUMENT = (
        "The task `do_run` must have a single TaskSpan argument."
    )

    def __init__(self, fast_api_app: FastAPI) -> None:
        self._fast_api_app = fast_api_app

    def register_task(
        self,
        task: Task[Input, Output],
        path: str,
    ) -> None:
        """Registers a task to your application.

        Registering a task will make it available as an endpoint.
        For technical reasons, your endpoint cannot have a `TaskSpan` as input.

        Args:
            task: The task you would like exposed.
            path: The path your exposed endpoint will have.

        Example:
            >>> import os
            >>> from aleph_alpha_client import Client
            >>> from fastapi import FastAPI
            >>> from intelligence_layer.core import Complete, IntelligenceApp

            >>> fast_api = FastAPI()
            >>> app = IntelligenceApp(fast_api)
            >>> aa_client = Client(os.getenv("AA_TOKEN"))
            >>> app.register_task(Complete(aa_client), "/complete")
        """
        input_type = self._verify_annotations(get_annotations(task.do_run))

        @self._fast_api_app.post(path)
        def task_route(input: Annotated[input_type, Body()]) -> Output:  # type: ignore
            return task.run(input, NoOpTracer())

    def _verify_annotations(self, annotations: dict[str, Any]) -> Any:
        annotations.pop("return", None)
        if len(annotations) != 2:
            raise RegisterTaskError(
                self.DO_RUN_NEEDS_TYPE_HINTS_FOR_INPUT_AND_TASK_SPAN
            )
        number_of_task_span_arguments = sum(
            1 for ty in annotations.values() if ty is TaskSpan
        )
        if number_of_task_span_arguments != 1:
            raise RegisterTaskError(self.DO_RUN_MUST_HAVE_SINGLE_TASKSPAN_ARGUMENT)
        input_type = next(ty for ty in annotations.values() if ty is not TaskSpan)
        return input_type

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
        task: Task[Input, Output],
        path: str,
        required_permissions: frozenset[str] = frozenset(),
    ) -> None:
        input_type = self._verify_annotations(get_annotations(task.do_run))

        @self._fast_api_app.post(path)
        def task_route(input: Annotated[input_type, Body()], allowed: Annotated[bool, Depends(partial(self._auth_service.get_permissions, required_permissions))]) -> Output:  # type: ignore
            if allowed:
                return task.run(input, NoOpTracer())
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="No permission rights",
                )
