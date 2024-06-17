from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Optional, cast

from intelligence_layer.core import InMemoryTracer, Output, PydanticSerializable
from intelligence_layer.core.tracer.tracer import Tracer
from intelligence_layer.evaluation.run.domain import ExampleOutput, RunOverview
from intelligence_layer.evaluation.run.run_repository import RunRepository


class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        self._example_outputs: dict[str, list[ExampleOutput[PydanticSerializable]]] = (
            defaultdict(list)
        )
        self._example_traces: dict[str, Tracer] = dict()
        self._run_overviews: dict[str, RunOverview] = dict()

    def store_run_overview(self, overview: RunOverview) -> None:
        self._run_overviews[overview.id] = overview
        if overview.id not in self._example_outputs:
            self._example_outputs[overview.id] = []

    def create_temporary_run_data(self, run_id: str) -> None: ...

    def delete_temporary_run_data(self, run_id: str) -> None: ...

    def temp_store_finished_example(self, run_id: str, example_id: str) -> None: ...

    def unfinished_examples(self) -> dict[str, Sequence[str]]: ...

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
        if run_id not in self._example_outputs:
            raise ValueError(f"Repository does not contain a run with id: {run_id}")

        if run_id not in self._example_outputs:
            return None

        for example_output in self._example_outputs[run_id]:
            if example_output.example_id == example_id:
                return cast(ExampleOutput[Output], example_output)
        return None

    def example_tracer(self, run_id: str, example_id: str) -> Optional[Tracer]:
        return self._example_traces.get(f"{run_id}/{example_id}")

    def create_tracer_for_example(self, run_id: str, example_id: str) -> Tracer:
        tracer = InMemoryTracer()
        self._example_traces[f"{run_id}/{example_id}"] = tracer
        return tracer

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output]]:
        if run_id not in self._run_overviews:
            raise ValueError(f"Repository does not contain a run with id: {run_id}")

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
