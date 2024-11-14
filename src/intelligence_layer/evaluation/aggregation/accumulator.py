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
        """Responsible for accumulating values.

        Args:
            value: the value to add
        Returns:
             nothing
        """
        ...

    @abstractmethod
    def extract(self) -> Output:
        """Accumulates the final result.

        Returns:
           float: 0.0 if no values were added before, else the mean
        """
        ...


class MeanAccumulator(Accumulator[float, float]):
    def __init__(self) -> None:
        self._n = 0
        self._acc = 0.0
        self._squares_acc = 0.0  # Sum of squares of the values

    def add(self, value: float) -> None:
        self._n += 1
        self._acc += value
        self._squares_acc += value**2

    def extract(self) -> float:
        """Accumulates the mean.

        :return: 0.0 if no values were added before, else the mean
        """
        return 0.0 if self._n == 0 else self._acc / self._n

    def standard_deviation(self) -> float:
        """Calculates the standard deviation."""
        if self._n == 0:
            return 0.0
        mean = self.extract()
        variance = (self._squares_acc / self._n) - (mean**2)
        return variance**0.5

    def standard_error(self) -> float:
        """Calculates the standard error of the mean."""
        if self._n <= 1:
            return 0.0
        return self.standard_deviation() / (self._n**0.5)
