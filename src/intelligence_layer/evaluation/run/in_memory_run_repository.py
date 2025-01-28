import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Optional, cast

from pydantic import TypeAdapter

from intelligence_layer.core import InMemoryTracer, Output, PydanticSerializable
from intelligence_layer.core.tracer.tracer import Tracer
from intelligence_layer.evaluation.run.domain import (
    ExampleOutput,
    FailedExampleRun,
    RunOverview,
)
from intelligence_layer.evaluation.run.run_repository import RecoveryData, RunRepository


class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        super().__init__()
        self._example_outputs: dict[str, list[ExampleOutput[PydanticSerializable]]] = (
            defaultdict(list)
        )
        self._example_traces: dict[str, Tracer] = dict()
        self._run_overviews: dict[str, RunOverview] = dict()
        self._recovery_data: dict[str, RecoveryData] = dict()

    def store_run_overview(self, overview: RunOverview) -> None:
        self._run_overviews[overview.id] = overview
        if overview.id not in self._example_outputs:
            self._example_outputs[overview.id] = []

    def _create_temporary_run_data(self, tmp_hash: str, run_id: str) -> None:
        self._recovery_data[tmp_hash] = RecoveryData(run_id=run_id)

    def _delete_temporary_run_data(self, tmp_hash: str) -> None:
        del self._recovery_data[tmp_hash]

    def _temp_store_finished_example(self, tmp_hash: str, example_id: str) -> None:
        self._recovery_data[tmp_hash].finished_examples.append(example_id)

    def finished_examples(self, tmp_hash: str) -> Optional[RecoveryData]:
        if tmp_hash in self._recovery_data:
            return self._recovery_data[tmp_hash]
        else:
            return None

    def run_overview(self, run_id: str) -> Optional[RunOverview]:
        return self._run_overviews.get(run_id, None)

    def run_overview_ids(self) -> Sequence[str]:
        return sorted(self._run_overviews.keys())

    def store_example_output(self, example_output: ExampleOutput[Output]) -> None:
        self._example_outputs[example_output.run_id].append(
            cast(ExampleOutput[PydanticSerializable], example_output)
        )

    @staticmethod
    def _convert_to_type(data: Any, desired_type: type[Output]) -> Output:
        if type(data) is desired_type:
            return data
        else:
            return TypeAdapter(desired_type).validate_python(data)

    def _generate_output_from_internal_output(
        self,
        internal_output: ExampleOutput[PydanticSerializable],
        output_type: type[Output],
    ) -> ExampleOutput[Output] | ExampleOutput[FailedExampleRun]:
        if (
            type(internal_output.output) is output_type
            or type(internal_output.output) is FailedExampleRun
        ):
            return internal_output  # type: ignore
        return ExampleOutput[Output](
            run_id=internal_output.run_id,
            example_id=internal_output.example_id,
            output=self._convert_to_type(internal_output.output, output_type),
        )

    def example_output(
        self, run_id: str, example_id: str, output_type: type[Output]
    ) -> Optional[ExampleOutput[Output] | ExampleOutput[FailedExampleRun]]:
        if run_id not in self._example_outputs:
            warnings.warn(
                f'Repository does not contain a run with id: "{run_id}"', UserWarning
            )
            return None

        for example_output in self._example_outputs[run_id]:
            if example_output.example_id == example_id:
                return self._generate_output_from_internal_output(
                    example_output, output_type
                )
        return None

    def example_outputs(
        self, run_id: str, output_type: type[Output]
    ) -> Iterable[ExampleOutput[Output] | ExampleOutput[FailedExampleRun]]:
        if run_id not in self._example_outputs and run_id not in self._run_overviews:
            warnings.warn(
                f'Repository does not contain a run with id: "{run_id}"', UserWarning
            )
            return []

        return (
            self._generate_output_from_internal_output(example_output, output_type)
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

    def example_tracer(self, run_id: str, example_id: str) -> Optional[Tracer]:
        return self._example_traces.get(f"{run_id}/{example_id}")

    def create_tracer_for_example(self, run_id: str, example_id: str) -> Tracer:
        tracer = InMemoryTracer()
        self._example_traces[f"{run_id}/{example_id}"] = tracer
        return tracer
