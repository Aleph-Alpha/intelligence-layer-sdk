from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Iterable, Optional, Sequence, TypeVar, final

from pydantic import BaseModel

from intelligence_layer.core.tracer import PydanticSerializable, TaskSpan, Tracer


class Token(BaseModel):
    """A token class containing it's id and the raw token.

    This is used instead of the Aleph Alpha client Token class since this one is serializable,
    while the one from the client is not.
    """

    token: str
    token_id: int


Input = TypeVar("Input", bound=PydanticSerializable)
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output", bound=PydanticSerializable)
"""Interface of the output returned by the task."""


MAX_CONCURRENCY = 20
global_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)


class Task(ABC, Generic[Input, Output]):
    """Base task interface. This may consist of several sub-tasks to accomplish the given task.

    Generics:
        Input: Interface to be passed to the task with all data needed to run the process.
            Ideally, these are specified in terms related to the use-case, rather than lower-level
            configuration options.

        Output: Interface of the output returned by the task.
    """

    @abstractmethod
    def do_run(self, input: Input, task_span: TaskSpan) -> Output:
        """The implementation for this use case.

        This takes an input and runs the implementation to generate an output.
        It takes a `Span` for tracing of the process.
        The Input and Output are logged by default.

        Args:
            input: Generic input defined by the task implementation
            span: The `Span` used for tracing.
        Returns:
            Generic output defined by the task implementation.
        """
        ...

    @final
    def run(self, input: Input, tracer: Tracer, id: Optional[str] = None) -> Output:
        """Executes the implementation of `do_run` for this use case.

        This takes an input and runs the implementation to generate an output.
        It takes a `Tracer` for tracing of the process.
        The Input and Output are logged by default.

        Args:
            input: Generic input defined by the task implementation
            tracer: The `Tracer` used for tracing.
        Returns:
            Generic output defined by the task implementation.
        """
        with tracer.task_span(type(self).__name__, input, id=id) as task_span:
            output = self.do_run(input, task_span)
            task_span.record_output(output)
            return output

    @final
    def run_concurrently(
        self,
        inputs: Iterable[Input],
        tracer: Tracer,
        concurrency_limit: int = MAX_CONCURRENCY,
        id: Optional[str] = None,
    ) -> Sequence[Output]:
        """Executes multiple processes of this task concurrently.

        Each provided input is potentially executed concurrently to the others. There is a global limit
        on the number of concurrently executed tasks that is shared by all tasks of all types.

        Args:
            inputs: The inputs that are potentially processed concurrently.
            tracer: The tracer passed on the `run` method when executing a task.
            concurrency_limit: An optional additional limit for the number of concurrently executed task for
                this method call. This can be used to prevent queue-full or similar error of downstream APIs
                when the global concurrency limit is too high for a certain task.
        Returns:
            The Outputs generated by calling `run` for each given Input.
            The order of Outputs corresponds to the order of the Inputs.
        """

        with tracer.span(f"Concurrent {type(self).__name__} tasks", id=id) as span:
            with ThreadPoolExecutor(
                max_workers=min(concurrency_limit, MAX_CONCURRENCY)
            ) as executor:
                return list(executor.map(lambda input: self.run(input, span), inputs))
