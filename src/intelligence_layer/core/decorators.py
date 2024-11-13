"""Module for decorators used in the intelligence layer."""

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, get_type_hints

from pydantic import BaseModel

from intelligence_layer.core.tracer.tracer import (
    NoOpTracer,
    Tracer,
)

T = TypeVar("T")
P = ParamSpec("P")


class SerializableInput(BaseModel):
    """Pydantic model for serializing input data."""

    args: dict[str, Any]
    kwargs: dict[str, Any]


def trace(func: Callable[P, T]) -> Callable[P, T]:
    """A decorator to trace the execution of a method in a Task subclass.

    This decorator wraps the method execution with tracing logic, creating a task span and recording the output.
    It retrieves the tracer from the Task instance and uses the method name as the task name.

    Args:
        func: The method to be traced.

    Returns:
        Callable: A decorator that traces the function execution.
    """

    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        """Wrapper function to execute the original function with tracing.

        Args:
            self: The instance of the Task subclass.
            *args: Positional arguments passed to the original function.
            **kwargs: Keyword arguments passed to the original function.

        Returns:
            PydanticSerializable: The output of the original function.
        """
        tracer: Tracer = getattr(self, "_tracer", NoOpTracer())
        name = func.__name__

        arg_names = list(get_type_hints(func).keys())
        args_dict = {arg_names[i]: arg for i, arg in enumerate(args)}
        input_data = SerializableInput(args=args_dict, kwargs=kwargs)

        if self.current_task_span is None:
            self.current_task_span = tracer.task_span(name, input_data)
        else:
            self.current_task_span = self.current_task_span.task_span(name, input_data)

        with self.current_task_span as task_span:
            output: T = func(self, *args, **kwargs)
            task_span.record_output(
                SerializableInput(args={"output": output}, kwargs={})
            )
            return output

    return wrapper  # type: ignore
