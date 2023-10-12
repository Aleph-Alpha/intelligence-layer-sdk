from abc import ABC, abstractmethod
import functools
from typing import Callable, Sequence, Any
from pydantic import BaseModel
from intelligence_layer.task import DebugLogger


class SearchResult(BaseModel):
    score: float
    chunk: str


class BaseRetriever(ABC):
    # def __init_subclass__(cls, **kwargs: Any) -> None:
    #     """Decorates run method to auto log input and output for the task"""
    #     super().__init_subclass__(**kwargs)

    #     def log_run_input_output(
    #         func: Callable[[str, DebugLogger, int], Sequence[SearchResult]]
    #     ) -> Callable[[str, DebugLogger, int], Sequence[SearchResult]]:
    #         @functools.wraps(func)
    #         def inner(
    #             self: BaseRetriever,
    #             query: str,
    #             logger: DebugLogger,
    #             *,
    #             k: int
    #         ) -> Sequence[SearchResult]:
    #             logger.log("Query", query)
    #             logger.log("k", k)
    #             output = func(self, query=query, logger=logger, k=k)
    #             logger.log("Output", output)
    #             return output

    #         return inner

    #     cls.get_relevant_documents_with_scores = log_run_input_output(cls.get_relevant_documents_with_scores)  # type: ignore

    @abstractmethod
    def get_relevant_documents_with_scores(
        self, query: str, logger: DebugLogger, *, k: int
    ) -> Sequence[SearchResult]:
        pass

    @abstractmethod
    def add_documents(self, texts: Sequence[str]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
