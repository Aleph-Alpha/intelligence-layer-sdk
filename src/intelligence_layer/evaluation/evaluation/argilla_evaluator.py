from abc import ABC, abstractmethod

from intelligence_layer.connectors.argilla.argilla_client import ArgillaEvaluation
from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation.base_logic import EvaluationLogic
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.domain import (
    Example,
    ExpectedOutput,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    ArgillaEvaluationRepository,
    RecordDataSequence,
)
from intelligence_layer.evaluation.evaluation.evaluator import Evaluator
from intelligence_layer.evaluation.run.run_repository import RunRepository


class ArgillaEvaluationLogic(
    EvaluationLogic[Input, Output, ExpectedOutput, RecordDataSequence], ABC
):
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> RecordDataSequence:
        return self._to_record(example, *output)

    @abstractmethod
    def _to_record(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> RecordDataSequence:
        """This method is responsible for translating the `Example` and `Output` of the task to :class:`RecordData`


        Args:
            example: The example to be translated.
            output: The output of the example that was run.
        """
        ...


class ArgillaEvaluator(
    Evaluator[Input, Output, ExpectedOutput, ArgillaEvaluation],
    ABC,
):
    """Evaluator used to integrate with Argilla (https://github.com/argilla-io/argilla).

    Use this evaluator if you would like to easily do human eval.
    This evaluator runs a dataset and sends the input and output to Argilla to be evaluated.

     Arguments:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        description: Human-readable description for the evaluator.
        evaluation_logic: The logic to use for evaluation.

    Generics:
        Input: Interface to be passed to the :class:`Task` that shall be evaluated.
        Output: Type of the output of the :class:`Task` to be evaluated.
        ExpectedOutput: Output that is expected from the run with the supplied input.
        ArgillaEvaluation: Interface of the metrics that come from the Argilla task`.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: ArgillaEvaluationRepository,
        description: str,
        evaluation_logic: ArgillaEvaluationLogic[Input, Output, ExpectedOutput],
    ) -> None:
        super().__init__(
            dataset_repository,
            run_repository,
            evaluation_repository,
            description,
            evaluation_logic,  # type: ignore
        )

    def evaluation_type(self) -> type[ArgillaEvaluation]:  # type: ignore
        return ArgillaEvaluation
