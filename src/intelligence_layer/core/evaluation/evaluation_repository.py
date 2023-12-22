from collections import defaultdict
from json import loads
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO, cast
from uuid import UUID

from fsspec import AbstractFileSystem  # type: ignore
from fsspec.implementations.local import LocalFileSystem  # type: ignore
from huggingface_hub import HfFileSystem  # type: ignore
from pydantic import BaseModel, Field

from intelligence_layer.core.evaluation.domain import (
    Evaluation,
    ExampleEvaluation,
    ExampleOutput,
    ExampleTrace,
    FailedExampleEvaluation,
    PartialEvaluationOverview,
    RunOverview,
    TaskSpanTrace,
)
from intelligence_layer.core.evaluation.evaluator import (
    EvaluationOverviewType,
    EvaluationRepository,
)
from intelligence_layer.core.task import Output
from intelligence_layer.core.tracer import (
    EndSpan,
    EndTask,
    FileTracer,
    InMemorySpan,
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    LogEntry,
    LogLine,
    PlainEntry,
    PydanticSerializable,
    StartSpan,
    StartTask,
    Tracer,
)


class SerializedExampleEvaluation(BaseModel):
    """A json-serialized evaluation of a single example in a dataset.

    Attributes:
        eval_id: Identifier of the run the evaluated example belongs to.
        example_id: Unique identifier of the example this evaluation was created for.
        is_exception: qill be `True` if an exception occurred during evaluation.
        json_result: The actrual serialized evaluation result.
    """

    eval_id: str
    example_id: str
    is_exception: bool
    json_result: str

    @classmethod
    def from_example_result(
        cls, result: ExampleEvaluation[Evaluation]
    ) -> "SerializedExampleEvaluation":
        return cls(
            eval_id=result.eval_id,
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, FailedExampleEvaluation),
            example_id=result.example_id,
        )

    def to_example_result(
        self, evaluation_type: type[Evaluation]
    ) -> ExampleEvaluation[Evaluation]:
        if self.is_exception:
            return ExampleEvaluation(
                eval_id=self.eval_id,
                example_id=self.example_id,
                result=FailedExampleEvaluation.model_validate_json(self.json_result),
            )
        else:
            return ExampleEvaluation(
                eval_id=self.eval_id,
                example_id=self.example_id,
                result=evaluation_type.model_validate_json(self.json_result),
            )


class FileSystemEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in json-files on Hugging Face."""

    def __init__(self, fs: AbstractFileSystem, root_directory: str) -> None:
        assert root_directory[-1] != "/"
        self._fs: HfFileSystem = fs
        self._root_directory = root_directory

    def _run_root_directory(self) -> str:
        path = self._root_directory + "/" + "runs"
        self._fs.makedirs(path, exist_ok=True)
        return path

    def _run_directory(self, run_id: str) -> str:
        path = self._run_root_directory() + "/" + run_id
        self._fs.makedirs(path, exist_ok=True)
        return path

    def _output_directory(self, run_id: str) -> str:
        path = self._run_directory(run_id) + "/" + "output"
        self._fs.makedirs(path, exist_ok=True)
        return path

    def _trace_directory(self, run_id: str) -> str:
        path = self._run_directory(run_id) + "/" + "trace"
        self._fs.makedirs(path, exist_ok=True)
        return path

    def _eval_root_directory(self) -> str:
        path = self._root_directory + "/" + "evals"
        self._fs.makedirs(path, exist_ok=True)
        return path

    def _eval_directory(self, eval_id: str) -> str:
        path = self._eval_root_directory() + "/" + eval_id
        self._fs.makedirs(path, exist_ok=True)
        return path

    def _example_output_path(self, run_id: str, example_id: str) -> str:
        return (self._output_directory(run_id) + "/" + example_id) + ".json"

    def _example_trace_path(self, run_id: str, example_id: str) -> str:
        return (self._trace_directory(run_id) + "/" + example_id) + ".jsonl"

    def _example_result_path(self, eval_id: str, example_id: str) -> str:
        return (self._eval_directory(eval_id) + "/" + example_id) + ".json"

    def _evaluation_run_overview_path(self, eval_id: str) -> str:
        return self._eval_directory(eval_id) + ".json"

    def _run_overview_path(self, run_id: str) -> str:
        return self._run_directory(run_id) + ".json"

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        serialized_result = JsonSerializer(root=example_output)
        self._fs.write_text(
            self._example_output_path(example_output.run_id, example_output.example_id),
            serialized_result.model_dump_json(indent=2),
        )

    def example_output(
        self, run_id: str, example_id: str, output_type: type[Output]
    ) -> Optional[ExampleOutput[Output]]:
        file_path = self._example_output_path(run_id, example_id)
        if not self._fs.exists(file_path):
            return None
        content = self._fs.read_text(file_path)
        # Mypy does not accept dynamic types
        return ExampleOutput[output_type].model_validate_json(json_data=content)  # type: ignore

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        def load_example_output(
            path: Path,
        ) -> Optional[ExampleOutput[Output]]:
            id = path.with_suffix("").name
            return self.example_output(run_id, id, output_type)

        path = self._output_directory(run_id)
        output_files = self._fs.glob(path + "/*.json")
        return (
            example_output
            for example_output in sorted(
                (load_example_output(Path(file)) for file in output_files),
                key=lambda example_output: example_output.example_id
                if example_output
                else "",
            )
            if example_output
        )

    def example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        def fetch_result_from_file_name(
            path: Path,
        ) -> Optional[ExampleEvaluation[Evaluation]]:
            id = path.with_suffix("").name
            return self.example_evaluation(eval_id, id, evaluation_type)

        path = self._eval_directory(eval_id)
        logs = self._fs.glob(path + "/*.json")
        return [
            example_result
            for example_result in (
                fetch_result_from_file_name(Path(file)) for file in logs
            )
            if example_result
        ]

    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        results = self.example_evaluations(eval_id, evaluation_type)
        return [r for r in results if isinstance(r.result, FailedExampleEvaluation)]

    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleEvaluation[Evaluation]]:
        file_path = self._example_result_path(eval_id, example_id)
        if not self._fs.exists(file_path):
            return None
        content = self._fs.read_text(file_path)
        serialized_example = SerializedExampleEvaluation.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        file_path = self._example_trace_path(run_id, example_id)
        if not self._fs.exists(file_path):
            return None
        with self._fs.open(file_path, "r") as f:
            in_memory_tracer = _parse_log(f)
        trace = TaskSpanTrace.from_task_span(
            cast(InMemoryTaskSpan, in_memory_tracer.entries[0])
        )
        return ExampleTrace(run_id=run_id, example_id=example_id, trace=trace)

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        file_path = self._example_trace_path(run_id, example_id)
        # TODO handle closing the file, maybe Tracer interface should be context handler
        return FileTracer(self._fs.open(file_path, "w"))

    def store_example_evaluation(self, result: ExampleEvaluation[Evaluation]) -> None:
        serialized_result = SerializedExampleEvaluation.from_example_result(result)
        self._fs.write_text(
            self._example_result_path(result.eval_id, result.example_id),
            serialized_result.model_dump_json(indent=2),
        )

    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        file_path = self._evaluation_run_overview_path(eval_id)
        if not self._fs.exists(file_path):
            return None
        content = self._fs.read_text(file_path)
        return overview_type.model_validate_json(content)

    def store_evaluation_overview(self, overview: PartialEvaluationOverview) -> None:
        self._fs.write_text(
            self._evaluation_run_overview_path(overview.id),
            overview.model_dump_json(indent=2),
        )

    def run_overview(self, run_id: str) -> RunOverview | None:
        file_path = self._run_overview_path(run_id)
        if not self._fs.exists(file_path):
            return None
        content = self._fs.read_text(file_path)
        return RunOverview.model_validate_json(content)

    def store_run_overview(self, overview: RunOverview) -> None:
        self._fs.write_text(
            self._run_overview_path(overview.id), overview.model_dump_json(indent=2)
        )

    def run_ids(self) -> Sequence[str]:
        return [
            Path(path).parent.name
            for path in self._fs.glob(self._run_root_directory() + "/*/output")
        ]

    def eval_ids(self) -> Sequence[str]:
        return [
            path.stem for path in self._fs.glob(self._eval_root_directory() + "/*.json")
        ]


def _parse_log(f: TextIO) -> InMemoryTracer:
    tree_builder = TreeBuilder()

    for line in f:
        json_line = loads(line)
        log_line = LogLine.model_validate(json_line)
        if log_line.entry_type == StartTask.__name__:
            tree_builder.start_task(log_line)
        elif log_line.entry_type == EndTask.__name__:
            tree_builder.end_task(log_line)
        elif log_line.entry_type == StartSpan.__name__:
            tree_builder.start_span(log_line)
        elif log_line.entry_type == EndSpan.__name__:
            tree_builder.end_span(log_line)
        elif log_line.entry_type == PlainEntry.__name__:
            tree_builder.plain_entry(log_line)
        else:
            raise RuntimeError(f"Unexpected entry_type in {log_line}")
    assert tree_builder.root
    return tree_builder.root


class FileEvaluationRepository(FileSystemEvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in json-files.

    Args:
        root_directory: The folder where the json-files are stored. The folder (along with its parents)
            will be created if it does not exist yet.
    """

    def __init__(self, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        super().__init__(LocalFileSystem(), str(root_directory))


class TreeBuilder(BaseModel):
    root: InMemoryTracer = InMemoryTracer()
    tracers: dict[UUID, InMemoryTracer] = Field(default_factory=dict)
    tasks: dict[UUID, InMemoryTaskSpan] = Field(default_factory=dict)
    spans: dict[UUID, InMemorySpan] = Field(default_factory=dict)

    def start_task(self, log_line: LogLine) -> None:
        start_task = StartTask.model_validate(log_line.entry)
        child = InMemoryTaskSpan(
            name=start_task.name,
            input=start_task.input,
            start_timestamp=start_task.start,
        )
        self.tracers[start_task.uuid] = child
        self.tasks[start_task.uuid] = child
        self.tracers.get(start_task.parent, self.root).entries.append(child)

    def end_task(self, log_line: LogLine) -> None:
        end_task = EndTask.model_validate(log_line.entry)
        task_span = self.tasks[end_task.uuid]
        task_span.end_timestamp = end_task.end
        task_span.record_output(end_task.output)

    def start_span(self, log_line: LogLine) -> None:
        start_span = StartSpan.model_validate(log_line.entry)
        child = InMemorySpan(name=start_span.name, start_timestamp=start_span.start)
        self.tracers[start_span.uuid] = child
        self.spans[start_span.uuid] = child
        self.tracers.get(start_span.parent, self.root).entries.append(child)

    def end_span(self, log_line: LogLine) -> None:
        end_span = EndSpan.model_validate(log_line.entry)
        span = self.spans[end_span.uuid]
        span.end_timestamp = end_span.end

    def plain_entry(self, log_line: LogLine) -> None:
        plain_entry = PlainEntry.model_validate(log_line.entry)
        entry = LogEntry(
            message=plain_entry.message,
            value=plain_entry.value,
            timestamp=plain_entry.timestamp,
        )
        self.tracers[plain_entry.parent].entries.append(entry)


class InMemoryEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in memory.

    Preferred for quick testing or notebook use.
    """

    _example_outputs: dict[
        str, list[ExampleOutput[PydanticSerializable]]
    ] = defaultdict(list)
    _example_evaluations: dict[str, list[ExampleEvaluation[BaseModel]]] = defaultdict(
        list
    )
    _example_traces: dict[str, InMemoryTracer] = dict()
    _evaluation_run_overviews: dict[str, PartialEvaluationOverview] = dict()
    _run_overviews: dict[str, RunOverview] = dict()

    def run_ids(self) -> Sequence[str]:
        return list(self._example_outputs.keys())

    def eval_ids(self) -> Sequence[str]:
        return list(self._evaluation_run_overviews.keys())

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

    def example_evaluation(
        self, eval_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> ExampleEvaluation[Evaluation] | None:
        return next(
            (
                result
                for result in self.example_evaluations(eval_id, evaluation_type)
                if result.example_id == example_id
            ),
            None,
        )

    def store_example_evaluation(
        self, evaluation: ExampleEvaluation[Evaluation]
    ) -> None:
        self._example_evaluations[evaluation.eval_id].append(evaluation)

    def example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        return [
            cast(ExampleEvaluation[Evaluation], example_evaluation)
            for example_evaluation in self._example_evaluations[eval_id]
        ]

    def failed_example_evaluations(
        self, eval_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleEvaluation[Evaluation]]:
        results = self.example_evaluations(eval_id, evaluation_type)
        return [r for r in results if isinstance(r.result, FailedExampleEvaluation)]

    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        return cast(EvaluationOverviewType, self._evaluation_run_overviews[eval_id])

    def store_evaluation_overview(self, overview: PartialEvaluationOverview) -> None:
        self._evaluation_run_overviews[overview.id] = overview

    def run_overview(self, run_id: str) -> RunOverview | None:
        return self._run_overviews.get(run_id)

    def store_run_overview(self, overview: RunOverview) -> None:
        self._run_overviews[overview.id] = overview
