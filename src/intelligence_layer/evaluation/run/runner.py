from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import get_annotations
from itertools import islice
from typing import Any, Generic, Optional, cast
from uuid import uuid4

from dict_hash import dict_hash
from tqdm import tqdm

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import (
    CompositeTracer,
    Input,
    NoOpTracer,
    Output,
    Task,
    Tracer,
    utc_now,
)
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.infrastructure.repository_navigator import (
    RepositoryNavigator,
    RunLineage,
)
from intelligence_layer.evaluation.run.domain import (
    ExampleOutput,
    FailedExampleRun,
    RunOverview,
)
from intelligence_layer.evaluation.run.run_repository import RunRepository


class Runner(Generic[Input, Output]):
    def __init__(
        self,
        task: Task[Input, Output],
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        description: str,
    ) -> None:
        self._task = task
        self._run_repository = run_repository
        self._dataset_repository = dataset_repository
        self.description = description

    def output_type(self) -> type[Output]:
        """Returns the type of the evaluated task's output.

        This can be used to retrieve properly typed outputs of an evaluation run
        from a :class:`RunRepository`

        Returns:
            the type of the evaluated task's output.
        """
        try:
            output_type = get_annotations(self._task.do_run)["return"]
        except KeyError:
            raise TypeError(
                f"Task of type {type(self._task)} must have a type-hint for the return value of do_run to detect the output_type. "
                f"Alternatively overwrite output_type() in {type(self)}"
            ) from None
        return cast(type[Output], output_type)

    def input_type(self) -> type[Input]:
        try:
            input_type = get_annotations(self._task.do_run)["input"]
        except KeyError:
            raise TypeError(
                f"Task of type {type(self._task)} must have a type-hint for the input value of do_run to detect the input_type. "
                f"Alternatively overwrite input_type() in {type(self)}"
            ) from None
        return cast(type[Input], input_type)

    def _run_hash(self, dataset_id: str, run_description: str) -> str:
        return str(hash(dataset_id + self.description + run_description))

    def run_dataset(
        self,
        dataset_id: str,
        tracer: Optional[Tracer] = None,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
        max_workers: int = 10,
        description: Optional[str] = None,
        trace_examples_individually: bool = True,
        labels: Optional[set[str]] = None,
        metadata: Optional[SerializableDict] = None,
        resume_from_recovery_data: bool = False,
    ) -> RunOverview:
        """Generates all outputs for the provided dataset.

        Will run each :class:`Example` provided in the dataset through the :class:`Task`.

        Args:
            dataset_id: The id of the dataset to generate output for. Consists of examples, each
                with an :class:`Input` and an :class:`ExpectedOutput` (can be None).
            tracer: An optional :class:`Tracer` to trace all the runs from each example.
                Use `trace_examples_individually` to trace each example with a dedicated tracer individually.
            num_examples: An optional int to specify how many examples from the dataset should be run.
                Always the first n examples will be taken.
            abort_on_error: Flag to abort all run when an error occurs. Defaults to False.
            max_workers: Number of examples that can be evaluated concurrently. Defaults to 10.
            description: An optional description of the run. Defaults to None.
            trace_examples_individually: Flag to create individual tracers for each example. Defaults to True.
            labels: A list of labels for filtering. Defaults to an empty list.
            metadata: A dict for additional information about the run overview. Defaults to an empty dict.
            resume_from_recovery_data: Flag to resume if execution failed previously.

        Returns:
            An overview of the run. Outputs will not be returned but instead stored in the
            :class:`RunRepository` provided in the __init__.
        """
        if labels is None:
            labels = set()
        if metadata is None:
            metadata = dict()

        run_id = str(uuid4())
        tmp_hash = self._run_hash(dataset_id, description or "")

        recovery_data = self._run_repository.finished_examples(tmp_hash)
        finished_examples: frozenset[str] = frozenset()
        if recovery_data is not None and resume_from_recovery_data:
            run_id = recovery_data.run_id
            finished_examples = frozenset(recovery_data.finished_examples)
        else:
            self._run_repository.create_temporary_run_data(tmp_hash, run_id)

        examples = self._dataset_repository.examples(
            dataset_id,
            self.input_type(),
            Any,  # type: ignore
            examples_to_skip=finished_examples,
        )
        if examples is None:
            raise ValueError(f"Dataset with id {dataset_id} not found")
        if num_examples:
            examples = islice(examples, num_examples)

        def run(
            example: Example[Input, ExpectedOutput],
        ) -> None:
            if trace_examples_individually:
                example_tracer = self._run_repository.create_tracer_for_example(
                    run_id, example.id
                )
                if tracer:
                    example_tracer = CompositeTracer([example_tracer, tracer])
            elif tracer:
                example_tracer = tracer
            else:
                example_tracer = NoOpTracer()

            output: Output | FailedExampleRun
            try:
                output = self._task.run(example.input, example_tracer)
            except Exception as e:
                if abort_on_error:
                    raise e
                print(
                    f'FAILED RUN: example "{example.id}", {type(e).__qualname__}: "{e}"'
                )
                output = FailedExampleRun.from_exception(e)

            self._run_repository.store_example_output_parallel(
                tmp_hash,
                ExampleOutput[Output](
                    run_id=run_id, example_id=example.id, output=output
                ),
            )
            self._run_repository.temp_store_finished_example(tmp_hash, example.id)

        start = utc_now()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run, example) for example in examples]  # type: ignore
            with tqdm(total=len(futures)) as pbar:
                pbar.set_description("Running Task")
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
        self._run_repository.delete_temporary_run_data(tmp_hash)

        full_description = (
            self.description + " : " + description if description else self.description
        )

        successful = 0
        failed = 0
        for example_output in self._run_repository.example_outputs(
            run_id, self.output_type()
        ):
            if isinstance(example_output.output, FailedExampleRun):
                failed += 1
            else:
                successful += 1

        run_overview = RunOverview(
            dataset_id=dataset_id,
            id=run_id,
            start=start,
            end=utc_now(),
            failed_example_count=failed,
            successful_example_count=successful,
            description=full_description,
            labels=labels,
            metadata=metadata,
        )

        self._run_repository.store_run_overview(run_overview)
        return run_overview

    def failed_runs(
        self, run_id: str, expected_output_type: type[ExpectedOutput]
    ) -> Iterable[RunLineage[Input, ExpectedOutput, Output]]:
        """Returns the `RunLineage` objects for all failed example runs that belong to the given run ID.

        Args:
            run_id: The ID of the run overview
            expected_output_type: Type of output that the `Task` returned in :func:`Task.do_run`

        Returns:
            :class:`Iterable` of :class:`RunLineage`s.
        """
        failed_example_outputs = self._run_repository.failed_example_outputs(
            run_id, output_type=self.output_type()
        )
        lineages = (
            self.run_lineage(run_id, output.example_id, expected_output_type)
            for output in failed_example_outputs
        )
        return (lineage for lineage in lineages if lineage is not None)

    def run_lineages(
        self,
        run_id: str,
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[RunLineage[Input, ExpectedOutput, Output]]:
        """Wrapper for `RepositoryNavigator.run_lineages`.

        Args:
            run_id: The id of the run
            expected_output_type: The type of the expected output as defined by the :class:`Example`

        Returns:
            An iterator over all :class:`RunLineage`s for the given run id.
        """
        navigator = RepositoryNavigator(self._dataset_repository, self._run_repository)
        return navigator.run_lineages(
            run_id=run_id,
            input_type=self.input_type(),
            expected_output_type=expected_output_type,
            output_type=self.output_type(),
        )

    def run_lineage(
        self,
        run_id: str,
        example_id: str,
        expected_output_type: type[ExpectedOutput],
    ) -> RunLineage[Input, ExpectedOutput, Output] | None:
        """Wrapper for `RepositoryNavigator.run_lineage`.

        Args:
            run_id: The id of the run
            example_id: The id of the example of interest
            expected_output_type: The type of the expected output as defined by the :class:`Example`

        Returns:
            The :class:`RunLineage` for the given run id and example id, `None` if the example or an output for the example does not exist.
        """
        navigator = RepositoryNavigator(self._dataset_repository, self._run_repository)
        return navigator.run_lineage(
            run_id=run_id,
            example_id=example_id,
            input_type=self.input_type(),
            expected_output_type=expected_output_type,
            output_type=self.output_type(),
        )

    def run_is_already_computed(
        self,
        metadata: SerializableDict,
    ) -> bool:
        """Checks if a run with the given metadata has already been computed.

        Args:
            metadata: The metadata dictionary to check.

        Returns:
            True if a run with the same metadata has already been computed. False otherwise.
        """
        previous_runs = {
            dict_hash(run_overview.metadata)
            for run_overview in self._run_repository.run_overviews()
        }
        return dict_hash(metadata) in previous_runs
