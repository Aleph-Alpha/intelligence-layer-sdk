import json
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence, cast

from flatten_json import flatten, unflatten_list  # type: ignore
from wandb import Artifact, Table
from wandb.sdk.wandb_run import Run

from intelligence_layer.core.task import Output
from intelligence_layer.core.tracer import (
    FileTracer,
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    NoOpTracer,
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
    @abstractmethod
    def run_ids(self) -> Sequence[str]:
        """Returns the ids of all stored runs.

        Having the id of a run, its outputs can be retrieved with
        :meth:`EvaluationRepository.example_outputs`.

        Returns:
            The ids of all stored runs.
        """
        ...

    @abstractmethod
    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        """Returns all :class:`ExampleOutput` for a given run.

        Args:
            run_id: The unique identifier of the run.
            output_type: Type of output that the `Task` returned
                in :func:`Task.do_run`

        Returns:
            Iterable over all outputs.
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
    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        """Returns an :class:`ExampleTrace` for an example in a run.

        Args:
            run_id: The unique identifier of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
            example_output: The actual output.
        """
        ...

    @abstractmethod
    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        """Returns a :class:`Tracer` to trace an individual example run.

        Args:
            run_id: The unique identifier of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
        """
        ...

    @abstractmethod
    def run_overview(self, run_id: str) -> RunOverview | None:
        """Returns an :class:`RunOverview` of a given run by its id.

        Args:
            run_id: Identifier of the eval run to obtain the overview for.

        Returns:
            :class:`RunOverview` if one was found, `None` otherwise.
        """
        ...

    @abstractmethod
    def store_run_overview(self, overview: RunOverview) -> None:
        """Stores an :class:`RunOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """
        ...


class FileRunRepository(RunRepository, FileBasedRepository):
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

    def run_overview(self, run_id: str) -> RunOverview | None:
        file_path = self._run_overview_path(run_id)
        if not file_path.exists():
            return None
        content = self.read_utf8(file_path)
        return RunOverview.model_validate_json(content)

    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        file_path = self._example_trace_path(run_id, example_id)
        if not file_path.exists():
            return None
        in_memory_tracer = self._parse_log(file_path)
        trace = TaskSpanTrace.from_task_span(
            cast(InMemoryTaskSpan, in_memory_tracer.entries[0])
        )
        return ExampleTrace(run_id=run_id, example_id=example_id, trace=trace)

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

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        def load_example_output(
            path: Path,
        ) -> Optional[ExampleOutput[Output]]:
            id = path.with_suffix("").name
            return self.example_output(run_id, id, output_type)

        path = self._output_directory(run_id)
        output_files = path.glob("*.json")
        return (
            example_output
            for example_output in sorted(
                (load_example_output(file) for file in output_files),
                key=lambda example_output: (
                    example_output.example_id if example_output else ""
                ),
            )
            if example_output
        )

    def run_ids(self) -> Sequence[str]:
        return [
            path.parent.name for path in self._run_root_directory().glob("*/output")
        ]

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        file_path = self._example_trace_path(run_id, example_id)
        return FileTracer(file_path)

    def store_run_overview(self, overview: RunOverview) -> None:
        self.write_utf8(
            self._run_overview_path(overview.id), overview.model_dump_json(indent=2)
        )

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        serialized_result = JsonSerializer(root=example_output)
        self.write_utf8(
            self._example_output_path(example_output.run_id, example_output.example_id),
            serialized_result.model_dump_json(indent=2),
        )


class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        self._example_outputs: dict[str, list[ExampleOutput[PydanticSerializable]]] = (
            defaultdict(list)
        )
        self._example_traces: dict[str, InMemoryTracer] = dict()
        self._run_overviews: dict[str, RunOverview] = dict()

    def run_ids(self) -> Sequence[str]:
        return list(self._example_outputs.keys())

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        self._example_outputs[example_output.run_id].append(
            cast(ExampleOutput[PydanticSerializable], example_output)
        )

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

    def run_overview(self, run_id: str) -> RunOverview | None:
        return self._run_overviews.get(run_id)

    def store_run_overview(self, overview: RunOverview) -> None:
        self._run_overviews[overview.id] = overview


class WandbRunRepository(RunRepository):
    def __init__(self) -> None:
        self._example_outputs: dict[str, Table] = dict()
        self._run_overviews: dict[str, Table] = dict()
        self._run: Run | None = None
        self.team_name: str = "aleph-alpha-intelligence-layer-trial"

    def run_ids(self) -> Sequence[str]:
        """Returns the ids of all stored runs.

        Having the id of a run, its outputs can be retrieved with
        :meth:`EvaluationRepository.example_outputs`.

        Returns:
            The ids of all stored runs.
        """
        raise NotImplementedError

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        """Returns all :class:`ExampleOutput` for a given run.

        Args:
            run_id: The unique identifier of the run.
            output_type: Type of output that the `Task` returned
                in :func:`Task.do_run`

        Returns:
            Iterable over all outputs.
        """
        table = self._get_table(run_id, "example_outputs")
        return [ExampleOutput[output_type].model_validate_json(self._unflatten_table_row(row, table.columns)) for _, row in table.iterrows()]  # type: ignore

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        flat_example_output = flatten(
            json.loads(example_output.model_dump_json()), separator="|"
        )
        if example_output.run_id not in self._example_outputs:
            self._example_outputs[example_output.run_id] = Table(columns=list(flat_example_output.keys()))  # type: ignore
        self._example_outputs[example_output.run_id].add_data(  # type: ignore
            *flatten(
                json.loads(example_output.model_dump_json()), separator="|"
            ).values()
        )

    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        """Returns an :class:`ExampleTrace` for an example in a run.

        Args:
            run_id: The unique identifier of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
            example_output: The actual output.
        """
        pass

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        """Returns a :class:`Tracer` to trace an individual example run.

        Args:
            run_id: The unique identifier of the run.
            example_id: Example identifier, will match :class:`ExampleEvaluation` identifier.
        """
        return NoOpTracer()

    def run_overview(self, run_id: str) -> RunOverview | None:
        """Returns an :class:`RunOverview` of a given run by its id.

        Args:
            run_id: Identifier of the eval run to obtain the overview for.

        Returns:
            :class:`RunOverview` if one was found, `None` otherwise.
        """
        table = self._get_table(run_id, "run_overview")
        return RunOverview.model_validate_json(
            json_data=self._unflatten_table_row(next(table.iterrows())[1], table.columns)  # type: ignore
        )

    def store_run_overview(self, overview: RunOverview) -> None:
        """Stores an :class:`RunOverview` in the repository.

        Args:
            overview: The overview to be persisted.
        """

        flat_overview = flatten(json.loads(overview.model_dump_json()), separator="|")
        if overview.id not in self._run_overviews:
            self._run_overviews[overview.id] = Table(columns=list(flat_overview.keys()))  # type: ignore
        self._run_overviews[overview.id].add_data(*flat_overview.values())  # type: ignore

    def _unflatten_table_row(self, row: list[str], columns: str) -> str:
        return json.dumps(unflatten_list(dict(zip(columns, row)), separator="|"))

    # @lru_cache(maxsize=2) If we want the wandb lineage to work, we cannot cache the table
    def _get_table(self, artifact_id: str, name: str) -> Table:
        if self._run is None:
            raise ValueError(
                "The run has not been started, are you using a WandbEvaluator?"
            )
        artifact = self._run.use_artifact(
            f"{self.team_name}/{self._run.project_name()}/{artifact_id}:latest"
        )
        return artifact.get(name)  # type: ignore

    def start_run(self, run: Run) -> None:
        self._run = run

    def finish_run(self) -> None:
        self._run = None

    def sync_table(self, run_id: str) -> None:
        if self._run is None:
            raise ValueError(
                "The run has not been started, are you using a WandbRunner?"
            )
        artifact = Artifact(name=run_id, type="Run")
        artifact.add(self._example_outputs[run_id], name="example_outputs")
        artifact.add(self._run_overviews[run_id], name="run_overview")
        self._run.log_artifact(artifact)
