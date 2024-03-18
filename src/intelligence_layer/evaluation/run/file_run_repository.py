from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, cast

from fsspec.implementations.local import LocalFileSystem  # type: ignore

from intelligence_layer.core import (
    FileTracer,
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    Output,
    Tracer,
)
from intelligence_layer.evaluation.infrastructure.file_system_based_repository import (
    FileSystemBasedRepository,
)
from intelligence_layer.evaluation.run.domain import ExampleOutput, RunOverview
from intelligence_layer.evaluation.run.run_repository import RunRepository
from intelligence_layer.evaluation.run.trace import ExampleTrace, TaskSpanTrace


class FileSystemRunRepository(RunRepository, FileSystemBasedRepository):
    def store_run_overview(self, overview: RunOverview) -> None:
        self.write_utf8(
            self._run_overview_path(overview.id), overview.model_dump_json(indent=2)
        )

    def run_overview(self, run_id: str) -> Optional[RunOverview]:
        file_path = self._run_overview_path(run_id)
        if not self.exists(file_path):
            return None

        content = self.read_utf8(file_path)
        return RunOverview.model_validate_json(content)

    def run_overview_ids(self) -> Sequence[str]:
        return sorted(
            [
                Path(f["name"]).stem
                for f in self._file_system.ls(
                    self.path_to_str(self._run_root_directory()), detail=True
                )
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


class FileRunRepository(FileSystemRunRepository):
    def __init__(self, root_directory: Path) -> None:
        super().__init__(LocalFileSystem(), root_directory)

    @staticmethod
    def path_to_str(path: Path) -> str:
        return str(path)
