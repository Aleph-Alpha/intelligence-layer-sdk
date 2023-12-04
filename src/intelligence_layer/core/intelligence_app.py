from inspect import get_annotations
from typing import Annotated

from fastapi import Body, FastAPI
from uvicorn import run

from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import NoOpTracer, TaskSpan


class InvalidTaskError(TypeError):
    """Error raised when incorrectly initialising a task for an IntelligenceApp."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class IntelligenceApp:
    """The Intelligence App is an easy way to turn your tasks into an application.

    This app is used to quickly setup a FastAPI server with any registered tasks that you would like.
    By registering your tasks you can expose them as endpoints ready for usage.

    Args:
        fast_api_app: This is the FastAPI app the IntelligenceApp relies on for routing.
    """
    def __init__(self, fast_api_app: FastAPI) -> None:
        self._fast_api_app = fast_api_app

    def register_task(self, task: Task[Input, Output], path: str) -> None:
        """Registers a task to your application.

        Registering a task will make it available as an endpoint.
        For technical reasons, your endpoint cannot have a `TaskSpan` as input.

        Args:
            task: The task you would like exposed.
            path: The path your exposed endpoint will have.

        Example:
        >>> app = IntelligenceApp()
        >>> app.register_task(Complete())
        """
        annotations = get_annotations(task.do_run)
        if len(annotations) < 3:
            raise InvalidTaskError(
                "The task `do_run` method needs a type for its input, task_span and return value."
            )
        if not annotations.pop("return", None):
            raise InvalidTaskError(
                "The task `do_run` method needs a type for it's return value."
            )
        task_span_arguments = [ty for ty in annotations.values() if ty is TaskSpan]
        if len(task_span_arguments) >= 2:
            raise InvalidTaskError(
                "The task `do_run` method cannot have a `TaskSpan` type as input."
            )
        elif len(task_span_arguments) == 0:
            raise InvalidTaskError(
                "The task `do_run` method needs a `TaskSpan` type as its second argument."
            )
        input_type = next(
            (
                ty
                for ty in list(annotations.values()).__reversed__()
                if ty is not TaskSpan
            ),
            None,
        )
        assert input_type

        @self._fast_api_app.post(path)
        def task_route(input: Annotated[input_type, Body()]) -> Output:  # type: ignore
            return task.run(input, NoOpTracer())

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """This starts the application.
        
        Args:
            host: The base url where the application will be served on.
            port: The port the application will listen to.
        """
        run(self._fast_api_app, host=host, port=port)
