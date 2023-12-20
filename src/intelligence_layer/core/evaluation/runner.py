from concurrent.futures import ThreadPoolExecutor
from inspect import get_annotations
from typing import Generic, Optional, cast
from uuid import uuid4

from tqdm import tqdm

from intelligence_layer.core.evaluation.domain import (
    Example,
    ExampleOutput,
    ExpectedOutput,
    FailedExampleRun,
    RunOverview,
)
from intelligence_layer.core.evaluation.evaluator import (
    DatasetRepository,
    EvaluationRepository,
)
from intelligence_layer.core.task import Input, Output, Task
from intelligence_layer.core.tracer import CompositeTracer, Tracer, utc_now


class Runner(Generic[Input, Output]):
    def __init__(
        self,
        task: Task[Input, Output],
        evaluation_repository: EvaluationRepository,
        dataset_repository: DatasetRepository,
        identifier: str,
    ) -> None:
        self._task = task
        self._evaluation_repository = evaluation_repository
        self._dataset_repository = dataset_repository
        self.identifier = identifier

    def output_type(self) -> type[Output]:
        """Returns the type of the evaluated task's output.

        This can be used to retrieve properly typed outputs of an evaluation run
        from a :class:`EvaluationRepository`

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
        self, dataset_id: str, tracer: Optional[Tracer] = None
    ) -> RunOverview:
        """Generates all outputs for the provided dataset.

        Will run each :class:`Example` provided in the dataset through the :class:`Task`.

        Args:
            dataset_id: The id of the dataset to generate output for. Consists of examples, each
                with an :class:`Input` and an :class:`ExpectedOutput` (can be None).
            output: Output of the :class:`Task` that shall be evaluated

        Returns:
            An overview of the run. Outputs will not be returned but instead stored in the
            :class:`EvaluationRepository` provided in the __init__.
        """

        def run(
            example: Example[Input, ExpectedOutput]
        ) -> tuple[str, Output | FailedExampleRun]:
            evaluate_tracer = self._evaluation_repository.example_tracer(
                run_id, example.id
            )
            if tracer:
                evaluate_tracer = CompositeTracer([evaluate_tracer, tracer])
            try:
                return example.id, self._task.run(example.input, evaluate_tracer)
            except Exception as e:
                return example.id, FailedExampleRun.from_exception(e)

        examples = self._dataset_repository.examples_by_id(
            dataset_id, self.input_type(), self.output_type()
        )
        if examples is None:
            raise ValueError(f"Dataset with id {dataset_id} not found")
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
            self._evaluation_repository.store_example_output(
                run_id, ExampleOutput[Output](example_id=example_id, output=output)
            )

        run_overview = RunOverview(
            dataset_id=dataset_id,
            id=run_id,
            start=start,
            end=utc_now(),
            failed_example_count=failed_count,
            successful_example_count=successful_count,
            runner_id=self.identifier,
        )
        self._evaluation_repository.store_run_overview(run_overview)
        return run_overview
