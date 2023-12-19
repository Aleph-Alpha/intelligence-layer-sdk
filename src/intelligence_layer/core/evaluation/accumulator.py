from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")
Output = TypeVar("Output")


class Accumulator(ABC, Generic[T, Output]):
    """Used for incremental computation.

    For use cases with large amount of data where you don't want to have every value in memory at once, e.g. evaluation.
    """

    @abstractmethod
    def add(self, value: T) -> None:
        """Responsible for accumulating values

        :param value: the value to add
        :return: nothing
        """
        ...

    @abstractmethod
    def extract(self) -> Output:
        """Accumulates the final result

        :return: the result of this calculation
        """
        ...


class MeanAccumulator(Accumulator[float, float]):
    def __init__(self) -> None:
        self._n = 0
        self._acc = 0.0

    def add(self, value: float) -> None:
        self._n += 1
        self._acc += value

    def extract(self) -> float:
        """Accumulates the mean

        :return: 0.0 if no values were added before, else the mean
        """
        return 0.0 if self._n == 0 else self._acc / self._n
