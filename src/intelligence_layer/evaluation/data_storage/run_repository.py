from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, cast

from intelligence_layer.core import (
    FileTracer,
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    Output,
    PydanticSerializable,
    Tracer,
)
from intelligence_layer.evaluation.data_storage.utils import FileBasedRepository
from intelligence_layer.evaluation.domain import (
    ExampleOutput,
    ExampleTrace,
    RunOverview,
    TaskSpanTrace,
)


class RunRepository(ABC):
    """Base run repository interface.

    Provides methods to store and load run results of a run.
    """

    @abstractmethod
    def store_run_overview(self, overview: RunOverview) -> None:
        """Stores a :class:`RunOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...

    @abstractmethod
    def run_overview(self, run_id: str) -> Optional[RunOverview]:
        """Returns a :class:`RunOverview` of a given run by its ID.

        Args:
            run_id: Identifier of the eval run to obtain the overview for.

        Returns:
            :class:`RunOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        """Stores an individual :class:`ExampleOutput`.

        Args:
            example_output: The actual output.
        """
        ...

    @abstractmethod
    def example_output_ids(self) -> Sequence[str]:
        """Returns the sorted IDs of all stored :class:`ExampleOutput`s.

        Returns:
            The IDs of all stored runs.
        """
        ...

    @abstractmethod
    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        """Returns all :class:`ExampleOutput` for a given run sorted by their example ID.

        Args:
            run_id: The unique identifier of the run.
            output_type: Type of output that the `Task` returned
                in :func:`Task.do_run`

        Returns:
            Iterable over all outputs.
        """
        ...

    @abstractmethod
    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        """Returns an :class:`ExampleTrace` for an example in a run.

        Args:
            run_id: The unique ID of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` ID.
        """
        ...

    @abstractmethod
    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        """Returns a :class:`Tracer` to trace an individual example run.

        Args:
            run_id: The unique ID of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` ID.
        """
        ...


class FileRunRepository(RunRepository, FileBasedRepository):
    def store_run_overview(self, overview: RunOverview) -> None:
        self.write_utf8(
            self._run_overview_path(overview.id), overview.model_dump_json(indent=2)
        )

    def run_overview(self, run_id: str) -> Optional[RunOverview]:
        file_path = self._run_overview_path(run_id)
        if not file_path.exists():
            return None
        content = self.read_utf8(file_path)
        return RunOverview.model_validate_json(content)

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        serialized_result = JsonSerializer(root=example_output)
        self.write_utf8(
            self._example_output_path(example_output.run_id, example_output.example_id),
            serialized_result.model_dump_json(indent=2),
        )

    def example_output(
        self, run_id: str, example_id: str, output_type: type[Output]
    ) -> Optional[ExampleOutput[Output]]:
        file_path = self._example_output_path(run_id, example_id)
        if not file_path.exists():
            return None
        content = self.read_utf8(file_path)
        # Mypy does not accept dynamic types
        return ExampleOutput[output_type].model_validate_json(  # type: ignore
            json_data=content
        )

    def example_output_ids(self) -> Sequence[str]:
        return sorted(
            [path.parent.name for path in self._run_root_directory().glob("*/output")]
        )

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        def load_example_output(
            file_path: Path,
        ) -> Optional[ExampleOutput[Output]]:
            example_id = file_path.with_suffix("").name
            return self.example_output(run_id, example_id, output_type)

        path = self._output_directory(run_id)
        output_files = path.glob("*.json")
        example_output = [load_example_output(file) for file in output_files]
        return sorted(
            [
                example_output
                for example_output in example_output
                if example_output is not None
            ],
            key=lambda _example_output: _example_output.example_id,
        )

    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        file_path = self._example_trace_path(run_id, example_id)
        if not file_path.exists():
            return None
        in_memory_tracer = self._parse_log(file_path)
        trace = TaskSpanTrace.from_task_span(
            cast(InMemoryTaskSpan, in_memory_tracer.entries[0])
        )
        return ExampleTrace(run_id=run_id, example_id=example_id, trace=trace)

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        file_path = self._example_trace_path(run_id, example_id)
        return FileTracer(file_path)

    def _example_trace_path(self, run_id: str, example_id: str) -> Path:
        return (self._trace_directory(run_id) / example_id).with_suffix(".jsonl")

    def _run_root_directory(self) -> Path:
        path = self._root_directory / "runs"
        path.mkdir(exist_ok=True)
        return path

    def _run_directory(self, run_id: str) -> Path:
        path = self._run_root_directory() / run_id
        path.mkdir(exist_ok=True)
        return path

    def _trace_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "trace"

        path.mkdir(exist_ok=True)
        return path

    def _run_overview_path(self, run_id: str) -> Path:
        return self._run_directory(run_id).with_suffix(".json")

    def _output_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "output"

        path.mkdir(exist_ok=True)
        return path

    @staticmethod
    def _parse_log(log_path: Path) -> InMemoryTracer:
        return FileTracer(log_path).trace()

    def _example_output_path(self, run_id: str, example_id: str) -> Path:
        return (self._output_directory(run_id) / example_id).with_suffix(".json")


class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        self._example_outputs: dict[str, list[ExampleOutput[PydanticSerializable]]] = (
            defaultdict(list)
        )
        self._example_traces: dict[str, InMemoryTracer] = dict()
        self._run_overviews: dict[str, RunOverview] = dict()

    def store_run_overview(self, overview: RunOverview) -> None:
        self._run_overviews[overview.id] = overview

    def run_overview(self, run_id: str) -> Optional[RunOverview]:
        return self._run_overviews.get(run_id, None)

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        self._example_outputs[example_output.run_id].append(
            cast(ExampleOutput[PydanticSerializable], example_output)
        )

    def example_output_ids(self) -> Sequence[str]:
        return sorted(list(self._example_outputs.keys()))

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        return (
            cast(ExampleOutput[Output], example_output)
            for example_output in sorted(
                self._example_outputs[run_id],
                key=lambda example_output: example_output.example_id,
            )
        )

    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        tracer = self._example_traces.get(f"{run_id}/{example_id}")
        if tracer is None:
            return None
        assert tracer
        return ExampleTrace(
            run_id=run_id,
            example_id=example_id,
            trace=TaskSpanTrace.from_task_span(
                cast(InMemoryTaskSpan, tracer.entries[0])
            ),
        )

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        tracer = InMemoryTracer()
        self._example_traces[f"{run_id}/{example_id}"] = tracer
        return tracer
