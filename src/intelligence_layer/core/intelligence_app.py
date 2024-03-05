from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Annotated, Generic, TypeVar

from fastapi import Depends, HTTPException, Response, status

from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer.tracer import Tracer

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
