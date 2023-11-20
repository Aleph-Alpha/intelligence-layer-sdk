from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Iterable, Optional, Protocol, Sequence, TypeVar, final
from uuid import uuid4

from pydantic import BaseModel, Field
from tqdm import tqdm

from intelligence_layer.core.task import Input
from intelligence_layer.core.tracer import PydanticSerializable, Tracer

ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=PydanticSerializable)
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=PydanticSerializable)


class Example(BaseModel, Generic[Input, ExpectedOutput]):
    """Example case used for evaluations.

    Attributes:
        input: Input for the task. Has to be same type as the input for the task used.
        expected_output: The expected output from a given example run.
            This will be used by the evaluator to compare the received output with.
        ident: Identifier for the example, defaults to uuid.
    """

    input: Input
    expected_output: ExpectedOutput
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))


class Dataset(Protocol, Generic[Input, ExpectedOutput]):
    """A dataset of examples used for evaluation of a task.

    Attributes:
        name: This a human readable identifier for a dataset.
        examples: The actual examples that a task will be evaluated on.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def examples(self) -> Iterable[Example[Input, ExpectedOutput]]:
        ...


class SequenceDataset(BaseModel, Generic[Input, ExpectedOutput]):
    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]


class EvaluationException(BaseModel):
    error_message: str


class Evaluator(ABC, Generic[Input, ExpectedOutput, Evaluation, AggregatedEvaluation]):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    We suggest supplying a `Task` in the `__init__` method and running it in the `evaluate` method.

    Generics:
        Input: Interface to be passed to the task that shall be evaluated.
        ExpectedOutput: Output that is expected from the task run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated task.
        AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.
    """

    @abstractmethod
    def do_evaluate(
        self,
        input: Input,
        tracer: Tracer,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case.

        The implementation of this method is responsible for running a task (usually supplied by the __init__ method)
        and making any comparisons relevant to the evaluation.
        Based on the results, it should create an `Evaluation` class with all the metrics and return it.

        Args:
            input: Interface to be passed to the task that shall be evaluated.
            tracer: Ttracer used for tracing of tasks.
            expected_output: Output that is expected from the task run with the supplied input.
        Returns:
            Interface of the metrics that come from the evaluated task.
        """
        pass

    def evaluate(
        self,
        input: Input,
        tracer: Tracer,
        expected_output: ExpectedOutput,
    ) -> Evaluation | EvaluationException:
        try:
            return self.do_evaluate(input, tracer, expected_output)
        except Exception as e:
            return EvaluationException(error_message=str(e))

    @final
    def evaluate_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], tracer: Tracer
    ) -> AggregatedEvaluation:
        """Evaluates an entire datasets in a threaded manner and aggregates the results into an `AggregatedEvaluation`.

        This will call the `run` method for each example in the dataset.
        Finally, it will call the `aggregate` method and return the aggregated results.

        Args:
            dataset: Dataset that will be used to evaluate a task.
            tracer: tracer used for tracing.
        Returns:
            The aggregated results of an evaluation run with a dataset.
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = tqdm(
                executor.map(
                    lambda idx_example: self.evaluate(
                        idx_example.input,
                        tracer,
                        idx_example.expected_output,
                    ),
                    dataset.examples,
                ),
                desc="Evaluating",
            )
        # collect errors with debug log
        return self.aggregate(
            evaluation
            for evaluation in evaluations
            if not isinstance(evaluation, EvaluationException)
        )

    @abstractmethod
    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evalautions: The results from running `evaluate_dataset` with a task.
        Returns:
            The aggregated results of an evaluation run with a dataset.
        """
        pass
