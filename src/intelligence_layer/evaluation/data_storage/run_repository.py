from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, cast
from fsspec.implementations.local import LocalFileSystem  # type: ignore


from intelligence_layer.core import (
    FileTracer,
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    Output,
    PydanticSerializable,
    Tracer,
)
from intelligence_layer.evaluation.data_storage.utils import FileSystemBasedRepository
from intelligence_layer.evaluation.domain import (
    ExampleOutput,
    ExampleTrace,
    RunOverview,
    TaskSpanTrace,
)


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


class FileSystemRunRepository(RunRepository, FileSystemBasedRepository):
    def store_run_overview(self, overview: RunOverview) -> None:
        self.write_utf8(
            self._run_overview_path(overview.id), overview.model_dump_json(indent=2)
        )

    def run_overview(self, run_id: str) -> Optional[RunOverview]:
        file_path = self._run_overview_path(run_id)
        if file_path is None:
            return None

        content = self.read_utf8(file_path)
        return RunOverview.model_validate_json(content)

    def run_overview_ids(self) -> Sequence[str]:
        return sorted(
            [
                Path(f["name"]).stem
                for f in self._fs.ls(self.path_to_str(self._run_root_directory()), detail=True)
                if isinstance(f, Dict) and Path(f["name"]).suffix == ".json"
            ]
        )

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
        # mypy does not accept dynamic types
        return ExampleOutput[output_type].model_validate_json(  # type: ignore
            json_data=content
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

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        def load_example_output(
            file_path: Path,
        ) -> Optional[ExampleOutput[Output]]:
            example_id = file_path.with_suffix("").name
            return self.example_output(run_id, example_id, output_type)

        path = self._run_output_directory(run_id)
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

    def example_output_ids(self, run_id: str) -> Sequence[str]:
        return sorted(
            [path.stem for path in self._run_output_directory(run_id).glob("*.json")]
        )

    def _run_root_directory(self) -> Path:
        path = self._root_directory / "runs"
        path.mkdir(exist_ok=True)
        return path

    def _run_directory(self, run_id: str) -> Path:
        path = self._run_root_directory() / run_id
        path.mkdir(exist_ok=True)
        return path

    def _run_output_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "output"
        path.mkdir(exist_ok=True)
        return path

    def _run_overview_path(self, run_id: str) -> Path:
        return self._run_directory(run_id).with_suffix(".json")

    def _trace_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "trace"
        path.mkdir(exist_ok=True)
        return path

    def _example_trace_path(self, run_id: str, example_id: str) -> Path:
        return (self._trace_directory(run_id) / example_id).with_suffix(".jsonl")

    @staticmethod
    def _parse_log(log_path: Path) -> InMemoryTracer:
        return FileTracer(log_path).trace()

    def _example_output_path(self, run_id: str, example_id: str) -> Path:
        return (self._run_output_directory(run_id) / example_id).with_suffix(".json")


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

    def run_overview_ids(self) -> Sequence[str]:
        return sorted(self._run_overviews.keys())

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        self._example_outputs[example_output.run_id].append(
            cast(ExampleOutput[PydanticSerializable], example_output)
        )

    def example_output(
        self, run_id: str, example_id: str, output_type: type[Output]
    ) -> Optional[ExampleOutput[Output]]:
        if run_id not in self._example_outputs.keys():
            return None

        for example_output in self._example_outputs[run_id]:
            if example_output.example_id == example_id:
                return cast(ExampleOutput[Output], example_output)
        return None

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

    def example_output_ids(self, run_id: str) -> Sequence[str]:
        return sorted(
            [
                example_output.example_id
                for example_output in self._example_outputs[run_id]
            ]
        )

class FileRunRepository(FileSystemRunRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)


    @staticmethod
    def path_to_str(path: Path) -> str:
        return str(path)