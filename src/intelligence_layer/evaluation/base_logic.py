from abc import ABC, abstractmethod
from typing import Generic, Iterable, final

from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation.domain import (
    AggregatedEvaluation,
    Evaluation,
    Example,
    ExpectedOutput,
    SuccessfulExampleOutput,
)


class AggregationLogic(ABC, Generic[Evaluation, AggregatedEvaluation]):
    @abstractmethod
    def aggregate(self, evaluations: Iterable[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evaluations: The results from running `eval_and_aggregate_runs` with a :class:`Task`.

        Returns:
            The aggregated results of an evaluation run with a :class:`Dataset`.
        """
        ...


class EvaluationLogic(ABC, Generic[Input, Output, ExpectedOutput, Evaluation]):
    @abstractmethod
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> Evaluation:
        """Executes the evaluation for this specific example.

        Responsible for comparing the input & expected output of a task to the
        actually generated output.

        Args:
            example: Input data of :class:`Task` to produce the output.
            output: Output of the :class:`Task`.

        Returns:
            The metrics that come from the evaluated :class:`Task`.
        """
        pass


class SingleOutputEvaluationLogic(
    EvaluationLogic[Input, Output, ExpectedOutput, Evaluation]
):
    @final
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> Evaluation:
        assert len(output) == 1
        return self.do_evaluate_single_output(example, output[0].output)

    @abstractmethod
    def do_evaluate_single_output(
        self, example: Example[Input, ExpectedOutput], output: Output
    ) -> Evaluation:
        pass
