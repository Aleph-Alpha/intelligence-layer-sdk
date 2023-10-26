from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Optional, Sequence, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field
from tqdm import tqdm

from intelligence_layer.core.task import Input
from intelligence_layer.core.logger import PydanticSerializable, DebugLogger


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
    ident: Optional[str] = Field(default_factory=lambda: str(uuid4()))


class Dataset(BaseModel, Generic[Input, ExpectedOutput]):
    """A dataset of examples used for evaluation of a task.

    Attributes:
        name: This a human readable identifier for a dataset.
        examples: The actual examples that a task will be evaluated on.
    """

    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]


class Evaluator(ABC, Generic[Input, ExpectedOutput, Evaluation, AggregatedEvaluation]):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    Generics:
        Input: Interface to be passed to the task that shall be evaluated.
        ExpectedOutput: Output that is expected from the task run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated task.
        AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.

    We suggest supplying a `Task` in the `__init__` method and running it in the `evaluate` method.
    """

    @abstractmethod
    def evaluate(
        self,
        input: Input,
        logger: DebugLogger,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case."""
        pass

    def evaluate_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], logger: DebugLogger
    ) -> AggregatedEvaluation:
        """Evaluates an entire datasets in a threaded manner and aggregates the results into an `AggregatedEvaluation`."""
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = list(
                tqdm(
                    executor.map(
                        lambda idx_example: self.evaluate(
                            idx_example.input,
                            logger,
                            idx_example.expected_output,
                        ),
                        dataset.examples,
                    ),
                    total=len(dataset.examples),
                    desc="Evaluating",
                )
            )
        return self.aggregate(evaluations)

    @abstractmethod
    def aggregate(self, evaluations: Sequence[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`."""
        pass
