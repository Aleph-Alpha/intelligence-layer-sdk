from concurrent.futures import ThreadPoolExecutor
from inspect import get_annotations
from itertools import islice
from typing import Generic, Iterable, Optional, cast
from uuid import uuid4

from pydantic import JsonValue
from tqdm import tqdm

from intelligence_layer.core import (
    CompositeTracer,
    Input,
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
            )
        return cast(type[Output], output_type)

    def input_type(self) -> type[Input]:
        try:
            input_type = get_annotations(self._task.do_run)["input"]
        except KeyError:
            raise TypeError(
                f"Task of type {type(self._task)} must have a type-hint for the input value of do_run to detect the input_type. "
                f"Alternatively overwrite input_type() in {type(self)}"
            )
        return cast(type[Input], input_type)

    def run_dataset(
        self,
        dataset_id: str,
        tracer: Optional[Tracer] = None,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
    ) -> RunOverview:
        """Generates all outputs for the provided dataset.

        Will run each :class:`Example` provided in the dataset through the :class:`Task`.

        Args:
            dataset_id: The id of the dataset to generate output for. Consists of examples, each
                with an :class:`Input` and an :class:`ExpectedOutput` (can be None).
            tracer: An optional :class:`Tracer` to trace all the runs from each example
            num_examples: An optional int to specify how many examples from the dataset should be run.
                Always the first n examples will be taken.
            abort_on_error: Flag to abort all run when an error occurs. Defaults to False.

        Returns:
            An overview of the run. Outputs will not be returned but instead stored in the
            :class:`RunRepository` provided in the __init__.
        """

        def run(
            example: Example[Input, ExpectedOutput],
        ) -> tuple[str, Output | FailedExampleRun]:
            evaluate_tracer = self._run_repository.example_tracer(run_id, example.id)
            if tracer:
                evaluate_tracer = CompositeTracer([evaluate_tracer, tracer])
            try:
                return example.id, self._task.run(example.input, evaluate_tracer)
            except Exception as e:
                if abort_on_error:
                    raise e
                print(
                    f'FAILED RUN: example {example.id}, {type(e).__qualname__}: "{e}"'
                )
                return example.id, FailedExampleRun.from_exception(e)

        # mypy does not like union types

        examples = self._dataset_repository.examples(
            dataset_id,
            self.input_type(),
            JsonValue,  # type: ignore
        )
        if examples is None:
            raise ValueError(f"Dataset with id {dataset_id} not found")
        if num_examples:
            examples = islice(examples, num_examples)
        run_id = str(uuid4())
        start = utc_now()
        with ThreadPoolExecutor(max_workers=10) as executor:
            ids_and_outputs = tqdm(executor.map(run, examples), desc="Evaluating")

            failed_count = 0
            successful_count = 0
            for example_id, output in ids_and_outputs:
                if isinstance(output, FailedExampleRun):
                    failed_count += 1
                else:
                    successful_count += 1
                self._run_repository.store_example_output(
                    ExampleOutput[Output](
                        run_id=run_id, example_id=example_id, output=output
                    ),
                )
        run_overview = RunOverview(
            dataset_id=dataset_id,
            id=run_id,
            start=start,
            end=utc_now(),
            failed_example_count=failed_count,
            successful_example_count=successful_count,
            description=self.description,
        )
        self._run_repository.store_run_overview(run_overview)
        return run_overview

    def run_lineages(
        self,
        run_id: str,
        expected_output_type: type[ExpectedOutput],
    ) -> Iterable[RunLineage[Input, ExpectedOutput, Output]]:
        """Wrapper for `RepositoryNagivator.run_lineages`.

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
        """Wrapper for `RepositoryNagivator.run_lineage`.

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
