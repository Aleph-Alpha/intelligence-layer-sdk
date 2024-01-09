from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, cast

from pydantic import BaseModel

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
    FileTracer,
    InMemoryTaskSpan,
    InMemoryTracer,
    JsonSerializer,
    PydanticSerializable,
    Tracer,
)


def write_utf8(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


class FileEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in json-files.

    Args:
        root_directory: The folder where the json-files are stored. The folder (along with its parents)
            will be created if it does not exist yet.
    """

    def __init__(self, root_directory: Path) -> None:
        root_directory.mkdir(parents=True, exist_ok=True)
        self._root_directory = root_directory

    def _run_root_directory(self) -> Path:
        path = self._root_directory / "runs"
        path.mkdir(exist_ok=True)
        return path

    def _run_directory(self, run_id: str) -> Path:
        path = self._run_root_directory() / run_id
        path.mkdir(exist_ok=True)
        return path

    def _output_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "output"
        path.mkdir(exist_ok=True)
        return path

    def _trace_directory(self, run_id: str) -> Path:
        path = self._run_directory(run_id) / "trace"
        path.mkdir(exist_ok=True)
        return path

    def _eval_root_directory(self) -> Path:
        path = self._root_directory / "evals"
        path.mkdir(exist_ok=True)
        return path

    def _eval_directory(self, eval_id: str) -> Path:
        path = self._eval_root_directory() / eval_id
        path.mkdir(exist_ok=True)
        return path

    def _example_output_path(self, run_id: str, example_id: str) -> Path:
        return (self._output_directory(run_id) / example_id).with_suffix(".json")

    def _example_trace_path(self, run_id: str, example_id: str) -> Path:
        return (self._trace_directory(run_id) / example_id).with_suffix(".jsonl")

    def _example_result_path(self, eval_id: str, example_id: str) -> Path:
        return (self._eval_directory(eval_id) / example_id).with_suffix(".json")

    def _evaluation_run_overview_path(self, eval_id: str) -> Path:
        return self._eval_directory(eval_id).with_suffix(".json")

    def _run_overview_path(self, run_id: str) -> Path:
        return self._run_directory(run_id).with_suffix(".json")

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        serialized_result = JsonSerializer(root=example_output)
        write_utf8(
            self._example_output_path(example_output.run_id, example_output.example_id),
            serialized_result.model_dump_json(indent=2),
        )

    def example_output(
        self, run_id: str, example_id: str, output_type: type[Output]
    ) -> Optional[ExampleOutput[Output]]:
        file_path = self._example_output_path(run_id, example_id)
        if not file_path.exists():
            return None
        content = read_utf8(file_path)
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
        output_files = path.glob("*.json")
        return (
            example_output
            for example_output in sorted(
                (load_example_output(file) for file in output_files),
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
        logs = path.glob("*.json")
        return [
            example_result
            for example_result in (fetch_result_from_file_name(file) for file in logs)
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
        if not file_path.exists():
            return None
        content = read_utf8(file_path)
        serialized_example = SerializedExampleEvaluation.model_validate_json(content)
        return serialized_example.to_example_result(evaluation_type)

    def example_trace(self, run_id: str, example_id: str) -> Optional[ExampleTrace]:
        file_path = self._example_trace_path(run_id, example_id)
        if not file_path.exists():
            return None
        in_memory_tracer = _parse_log(file_path)
        trace = TaskSpanTrace.from_task_span(
            cast(InMemoryTaskSpan, in_memory_tracer.entries[0])
        )
        return ExampleTrace(run_id=run_id, example_id=example_id, trace=trace)

    def example_tracer(self, run_id: str, example_id: str) -> Tracer:
        file_path = self._example_trace_path(run_id, example_id)
        return FileTracer(file_path)

    def store_example_evaluation(self, result: ExampleEvaluation[Evaluation]) -> None:
        serialized_result = SerializedExampleEvaluation.from_example_result(result)
        write_utf8(
            self._example_result_path(result.eval_id, result.example_id),
            serialized_result.model_dump_json(indent=2),
        )

    def evaluation_overview(
        self, eval_id: str, overview_type: type[EvaluationOverviewType]
    ) -> EvaluationOverviewType | None:
        file_path = self._evaluation_run_overview_path(eval_id)
        if not file_path.exists():
            return None
        content = read_utf8(file_path)
        return overview_type.model_validate_json(content)

    def store_evaluation_overview(self, overview: PartialEvaluationOverview) -> None:
        write_utf8(
            self._evaluation_run_overview_path(overview.id),
            overview.model_dump_json(indent=2),
        )

    def run_overview(self, run_id: str) -> RunOverview | None:
        file_path = self._run_overview_path(run_id)
        if not file_path.exists():
            return None
        content = read_utf8(file_path)
        return RunOverview.model_validate_json(content)

    def store_run_overview(self, overview: RunOverview) -> None:
        write_utf8(
            self._run_overview_path(overview.id), overview.model_dump_json(indent=2)
        )

    def run_ids(self) -> Sequence[str]:
        return [
            path.parent.name for path in self._run_root_directory().glob("*/output")
        ]

    def eval_ids(self) -> Sequence[str]:
        return [path.stem for path in self._eval_root_directory().glob("*.json")]


def _parse_log(log_path: Path) -> InMemoryTracer:
    return FileTracer(log_path).trace()


class InMemoryEvaluationRepository(EvaluationRepository):
    """An :class:`EvaluationRepository` that stores evaluation results in memory.

    Preferred for quick testing or notebook use.
    """

    def __init__(self) -> None:
        self._example_outputs: dict[
            str, list[ExampleOutput[PydanticSerializable]]
        ] = defaultdict(list)
        self._example_evaluations: dict[
            str, list[ExampleEvaluation[BaseModel]]
        ] = defaultdict(list)
        self._example_traces: dict[str, InMemoryTracer] = dict()
        self._evaluation_run_overviews: dict[str, PartialEvaluationOverview] = dict()
        self._run_overviews: dict[str, RunOverview] = dict()

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
