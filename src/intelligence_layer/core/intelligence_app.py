from abc import ABC, abstractmethod
from inspect import get_annotations
from pprint import pprint
from typing import Annotated, Any, Callable, Sequence

from fastapi import Body, FastAPI

from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import NoOpTracer, TaskSpan


class IntelligenceApp:
    def __init__(self, fast_api_app: FastAPI) -> None:
        self.fast_api_app = fast_api_app

    def register_task(self, task: Task[Input, Output], path: str) -> None:
        annotations = get_annotations(task.do_run)
        annotations.pop("return", None)
        assert any(ty is TaskSpan for ty in annotations.values())
        input_type = next(
            (
                ty
                for ty in list(annotations.values()).__reversed__()
                if ty is not TaskSpan
            ),
            None,
        )
        assert input_type

        @self.fast_api_app.post(path)
        def task_route(input: Annotated[input_type, Body()]) -> Output:  # type: ignore
            print(f"{type(input)}: {input}")
            return task.run(input, NoOpTracer())


# class Authenticator(ABC):
#     @abstractmethod
#     def check_scopes(self, required_scopes: frozenset[str]) -> bool:
#         pass


# class OAuthAuthenticator(Authenticator):
#     def __init__(self, request: Any) -> None:
#         # extract user scopes from request
#         self.user_scopes: frozenset[str] = frozenset()
#         pass

#     def check_scopes(self, required_scopes: frozenset[str]) -> bool:
#         return required_scopes.issubset(self.user_scopes)


# class ILApp:
#     def __init__(self, app: FastAPI) -> None:
#         ...

#     def register(
#         self, task: Task[Input, Output], path: str, required_scopres={}
#     ) -> None:
#         ...

#     def register_with_auth(
#         self, task: Task[Input, Output], path: str, required_scopes: frozenset[str]
#     ) -> None:
#         ...

#     def serve(self) -> None:
#         ...

#     def authenticator(self, authenticator: Callable[[Any], Authenticator]) -> None:
#         ...


# class MyTask(Task[str, int]):
#     def do_run(self, input: str, task_span: TaskSpan) -> int:
#         return int(input)


# def main(argv: Sequence[str]) -> None:
#     app = ILApp(FastAPI())
#     # default path = task-name
#     app.register(
#         MyTask(), "/{task_name}?trace"
#     )  # POST input -> {"output": output, "trace": trace}
#     # trace? return from endpoint? (and/or send to service)
#     app.register_with_auth(MyTask(), "/mytask", frozenset({"my-task-permission"}))
#     app.authenticator(OAuthAuthenticator)
#     app.serve()


# # TODO?
# # - prototyp
# # - integrate auth
# # - automatically add feedback-route for each task, evaluate the output for a given input
# #   -> could be added to fine tuning dataset
# # - preconfigured routes
# # - production environment
