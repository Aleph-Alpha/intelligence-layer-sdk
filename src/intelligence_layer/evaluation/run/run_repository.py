from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

from intelligence_layer.core import Output, Tracer
from intelligence_layer.evaluation.run.domain import (
    ExampleOutput,
    FailedExampleRun,
    RunOverview,
)
from intelligence_layer.evaluation.run.trace import ExampleTrace


class RunRepository(ABC):
    """Base run repository interface.

    Provides methods to store and load run results: :class:`RunOverview` and :class:`ExampleOutput`.
    A :class:`RunOverview` is created from and is linked (by its ID) to multiple :class:`ExampleOutput`s
    representing results of a dataset.
    """

    @abstractmethod
    def store_run_overview(self, overview: RunOverview) -> None:
        """Stores a :class:`RunOverview`.

        Args:
            overview: The overview to be persisted.
        """
        ...

    @abstractmethod
    def run_overview(self, run_id: str) -> Optional[RunOverview]:
        """Returns a :class:`RunOverview` for the given ID.

        Args:
            run_id: ID of the run overview to retrieve.

        Returns:
            :class:`RunOverview` if it was found, `None` otherwise.
        """
        ...

    def run_overviews(self) -> Iterable[RunOverview]:
        """Returns all :class:`RunOverview`s sorted by their ID.

        Returns:
            :class:`Iterable` of :class:`RunOverview`s.
        """
        for run_id in self.run_overview_ids():
            run_overview = self.run_overview(run_id)
            if run_overview is not None:
                yield run_overview

    @abstractmethod
    def run_overview_ids(self) -> Sequence[str]:
        """Returns sorted IDs of all stored :class:`RunOverview`s.

        Returns:
            A :class:`Sequence` of the :class:`RunOverview` IDs.
        """
        ...

    @abstractmethod
    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        """Stores an :class:`ExampleOutput`.

        Args:
            example_output: The example output to be persisted.
        """
        ...

    @abstractmethod
    def example_output(
        self, run_id: str, example_id: str, output_type: type[Output]
    ) -> Optional[ExampleOutput[Output]]:
        """Returns :class:`ExampleOutput` for the given run ID and example ID.

        Args:
            run_id: The ID of the linked run overview.
            example_id: ID of the example to retrieve.
            output_type: Type of output that the `Task` returned in :func:`Task.do_run`

        Returns:
            class:`ExampleOutput` if it was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        """Returns an :class:`ExampleTrace` for the given run ID and example ID.

        Args:
            run_id: The ID of the linked run overview.
            example_id: ID of the example whose :class:`ExampleTrace` should be retrieved.

        Returns:
            An :class:`ExampleTrace` if it was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        """Returns a :class:`Tracer` for the given run ID and example ID.

        Args:
            run_id: The ID of the linked run overview.
            example_id: ID of the example whose :class:`Tracer` should be retrieved.

        Returns:
            A :.class:`Tracer`.
        """
        ...

    @abstractmethod
    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        """Returns all :class:`ExampleOutput` for a given run ID sorted by their example ID.

        Args:
            run_id: The ID of the run overview.
            output_type: Type of output that the `Task` returned in :func:`Task.do_run`

        Returns:
            :class:`Iterable` of :class:`ExampleOutput`s.
        """
        ...

    @abstractmethod
    def example_output_ids(self, run_id: str) -> Sequence[str]:
        """Returns the sorted IDs of all :class:`ExampleOutput`s for a given run ID.

        Args:
            run_id: The ID of the run overview.

        Returns:
            A :class:`Sequence` of all :class:`ExampleOutput` IDs.
        """
        ...

    def successful_example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        """Returns all :class:`ExampleOutput` for successful example runs with a given run-overview ID sorted by their example ID.

        Args:
            run_id: The ID of the run overview.
            output_type: Type of output that the `Task` returned in :func:`Task.do_run`

        Returns:
            :class:`Iterable` of :class:`ExampleOutput`s.
        """
        results = self.example_outputs(run_id, output_type)
        return (r for r in results if not isinstance(r.output, FailedExampleRun))

    def failed_example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        """Returns all :class:`ExampleOutput` for failed example runs with a given run-overview ID sorted by their example ID.

        Args:
            run_id: The ID of the run overview.
            output_type: Type of output that the `Task` returned in :func:`Task.do_run`

        Returns:
            :class:`Iterable` of :class:`ExampleOutput`s.
        """
        results = self.example_outputs(run_id, output_type)
        return (r for r in results if isinstance(r.output, FailedExampleRun))
