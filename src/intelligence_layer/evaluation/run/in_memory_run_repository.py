from collections import defaultdict
from typing import Iterable, Optional, Sequence, cast

from intelligence_layer.core import (
    InMemoryTaskSpan,
    InMemoryTracer,
    Output,
    PydanticSerializable,
    Tracer,
)
from intelligence_layer.evaluation.run.domain import ExampleOutput, RunOverview
from intelligence_layer.evaluation.run.run_repository import RunRepository
from intelligence_layer.evaluation.run.trace import ExampleTrace, TaskSpanTrace


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
