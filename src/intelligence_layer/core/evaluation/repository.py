from collections import defaultdict
from json import loads
from pathlib import Path
from typing import Optional, Sequence, cast
from uuid import UUID

from pydantic import BaseModel, Field

from intelligence_layer.core.evaluation.domain import (
    AggregatedEvaluation,
    Evaluation,
    EvaluationException,
    EvaluationRunOverview,
    ExampleResult,
    ExampleTrace,
    TaskSpanTrace,
)
from intelligence_layer.core.evaluation.evaluator import EvaluationRepository
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
    StartSpan,
    StartTask,
    Tracer,
)


class SerializedExampleResult(BaseModel):
    example_id: str
    is_exception: bool
    json_result: str

    @classmethod
    def from_example_result(
        cls, result: ExampleResult[Evaluation]
    ) -> "SerializedExampleResult":
        return cls(
            json_result=JsonSerializer(root=result.result).model_dump_json(),
            is_exception=isinstance(result.result, EvaluationException),
            example_id=result.example_id,
        )

    def to_example_result(
        self, evaluation_type: type[Evaluation]
    ) -> ExampleResult[Evaluation]:
        if self.is_exception:
            return ExampleResult(
                example_id=self.example_id,
                result=EvaluationException.model_validate_json(self.json_result),
            )
        else:
            return ExampleResult(
                example_id=self.example_id,
                result=evaluation_type.model_validate_json(self.json_result),
            )


class FileEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in json-files.

    Args:
        root_directory: The folder where the json-files are stored. The folder (along with its parents)
            will be created if it does not exist yet.
    """

    def __init__(self, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory

    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        def fetch_result_from_file_name(
            path: Path,
        ) -> Optional[ExampleResult[Evaluation]]:
            id = path.with_suffix("").name.removesuffix("_result")
            return self.evaluation_example_result(run_id, id, evaluation_type)

        path = self._root_directory / run_id
        logs = path.glob("*.json")
        return [
            example_result
            for example_result in (fetch_result_from_file_name(file) for file in logs)
            if example_result
        ]

    def failed_evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        results = self.evaluation_run_results(run_id, evaluation_type)
        return [r for r in results if isinstance(r.result, EvaluationException)]

    def evaluation_example_result(
        self, run_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> Optional[ExampleResult[Evaluation]]:
        file_path = (
            self._root_directory / run_id / f"{example_id}_result"
        ).with_suffix(".json")
        if not file_path.exists():
            return None
        content = file_path.read_text()
        serialized_example = SerializedExampleResult.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def evaluation_example_trace(
        self, run_id: str, example_id: str
    ) -> Optional[ExampleTrace]:
        file_path = (self._root_directory / run_id / f"{example_id}_trace").with_suffix(
            ".jsonl"
        )
        if not file_path.exists():
            return None
        in_memory_tracer = _parse_log(file_path)
        trace = TaskSpanTrace.from_task_span(
            cast(InMemoryTaskSpan, in_memory_tracer.entries[0])
        )
        return ExampleTrace(example_id=example_id, trace=trace)

    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        run_path = self._root_directory / run_id
        run_path.mkdir(exist_ok=True)
        serialized_result = SerializedExampleResult.from_example_result(result)
        (run_path / f"{result.example_id}_result").with_suffix(".json").write_text(
            serialized_result.model_dump_json(indent=2)
        )

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        run_dir = self._root_directory / run_id
        run_dir.mkdir(exist_ok=True)
        file_path = (run_dir / f"{example_id}_trace").with_suffix(".jsonl")
        return FileTracer(file_path)

    def evaluation_run_overview(
        self, run_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[EvaluationRunOverview[AggregatedEvaluation]]:
        file_path = (self._root_directory / run_id).with_suffix(".json")
        if not file_path.exists():
            return None
        content = file_path.read_text()
        # Mypy does not accept dynamic types
        return EvaluationRunOverview[aggregation_type].model_validate_json(content)  # type: ignore

    def store_evaluation_run_overview(
        self, overview: EvaluationRunOverview[AggregatedEvaluation]
    ) -> None:
        (self._root_directory / overview.id).with_suffix(".json").write_text(
            overview.model_dump_json(indent=2)
        )

    def run_ids(self) -> Sequence[str]:
        return [
            path.with_suffix("").name for path in self._root_directory.glob("*.json")
        ]


def _parse_log(log_path: Path) -> InMemoryTracer:
    tree_builder = TreeBuilder()
    with log_path.open("r") as f:
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
    _example_results: dict[str, list[str]] = defaultdict(list)
    _example_traces: dict[str, InMemoryTracer] = dict()

    _run_overviews: dict[str, str] = dict()

    def evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        result_jsons = self._example_results.get(run_id, [])
        return [
            SerializedExampleResult.model_validate_json(json_str).to_example_result(
                evaluation_type
            )
            for json_str in result_jsons
        ]

    def failed_evaluation_run_results(
        self, run_id: str, evaluation_type: type[Evaluation]
    ) -> Sequence[ExampleResult[Evaluation]]:
        results = self.evaluation_run_results(run_id, evaluation_type)
        return [r for r in results if isinstance(r.result, EvaluationException)]

    def evaluation_example_result(
        self, run_id: str, example_id: str, evaluation_type: type[Evaluation]
    ) -> ExampleResult[Evaluation] | None:
        return next(
            (
                result
                for result in self.evaluation_run_results(run_id, evaluation_type)
                if result.example_id == example_id
            ),
            None,
        )

    def evaluation_example_trace(
        self, run_id: str, example_id: str
    ) -> Optional[ExampleTrace]:
        tracer = self._example_traces.get(f"{run_id}/{example_id}")
        if tracer is None:
            return None
        assert tracer
        return ExampleTrace(
            example_id=example_id,
            trace=TaskSpanTrace.from_task_span(
                cast(InMemoryTaskSpan, tracer.entries[0])
            ),
        )

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        tracer = InMemoryTracer()
        self._example_traces[f"{run_id}/{example_id}"] = tracer
        return tracer

    def store_example_result(
        self, run_id: str, result: ExampleResult[Evaluation]
    ) -> None:
        json_result = SerializedExampleResult.from_example_result(result)
        self._example_results[run_id].append(json_result.model_dump_json())

    def store_evaluation_run_overview(
        self, overview: EvaluationRunOverview[AggregatedEvaluation]
    ) -> None:
        self._run_overviews[overview.id] = overview.model_dump_json()

    def evaluation_run_overview(
        self, run_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> EvaluationRunOverview[AggregatedEvaluation] | None:
        loaded_json = self._run_overviews.get(run_id)
        # mypy doesn't accept dynamic types as type parameter
        return (
            EvaluationRunOverview[aggregation_type].model_validate_json(loaded_json)  # type: ignore
            if loaded_json
            else None
        )

    def run_ids(self) -> Sequence[str]:
        return list(self._run_overviews.keys())
