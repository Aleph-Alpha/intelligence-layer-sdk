from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, final

from tqdm import tqdm

from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import Input, Output, utc_now
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.evaluation.evaluator.base_evaluator import (
    EvaluationLogicBase,
    EvaluatorBase,
)
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput
from intelligence_layer.evaluation.run.run_repository import RunRepository


class EvaluationLogic(
    ABC, EvaluationLogicBase[Input, Output, ExpectedOutput, Evaluation]
):
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
            *output: Output of the :class:`Task`.

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


class Evaluator(EvaluatorBase[Input, Output, ExpectedOutput, Evaluation]):
    """Evaluator designed for most evaluation tasks. Only supports synchronous evaluation.

    See the :class:`EvaluatorBase` for more information.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: EvaluationRepository,
        description: str,
        evaluation_logic: EvaluationLogic[Input, Output, ExpectedOutput, Evaluation],
    ) -> None:
        super().__init__(
            dataset_repository,
            run_repository,
            evaluation_repository,
            description,
            evaluation_logic,
        )
        self._evaluation_logic: EvaluationLogic[
            Input, Output, ExpectedOutput, Evaluation
        ]

    def evaluate_runs(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
        skip_example_on_any_failure: bool = True,
        description: Optional[str] = None,
        labels: Optional[set[str]] = None,
        metadata: Optional[SerializableDict] = None,
    ) -> EvaluationOverview:
        """Evaluates all generated outputs in the run.

        For each set of successful outputs in the referenced runs,
        :func:`EvaluationLogic.do_evaluate` is called and eval metrics are produced &
        stored in the provided :class:`EvaluationRepository`.

        Args:
            *run_ids: The runs to be evaluated. Each run is expected to have the same
                dataset as input (which implies their tasks have the same input-type)
                and their tasks have the same output-type. For each example in the
                dataset referenced by the runs the outputs of all runs are collected
                and if all of them were successful they are passed on to the implementation
                specific evaluation. The method compares all run of the provided ids to each other.
            num_examples: The number of examples which should be evaluated from the given runs.
                Always the first n runs stored in the evaluation repository. Defaults to None.
            abort_on_error: Flag to abort all evaluations when an error occurs. Defaults to False.
            skip_example_on_any_failure: Flag to skip evaluation on any example for which at least one run fails. Defaults to True.
            description: Optional description of the evaluation. Defaults to None.
            labels: A list of labels for filtering. Defaults to an empty list.
            metadata: A dict for additional information about the evaluation overview. Defaults to an empty dict.

        Returns:
            EvaluationOverview: An overview of the evaluation. Individual :class:`Evaluation`s will not be
            returned but instead stored in the :class:`EvaluationRepository` provided in the
            __init__.
        """
        if metadata is None:
            metadata = dict()
        if labels is None:
            labels = set()
        start = utc_now()
        run_overviews = self._load_run_overviews(*run_ids)
        eval_id = self._evaluation_repository.initialize_evaluation()

        with ThreadPoolExecutor(max_workers=10) as executor:
            example_evaluations = list(  # the list is needed to consume the iterator returned from the executor.map
                tqdm(
                    executor.map(
                        lambda args: self.evaluate(
                            args[0], eval_id, abort_on_error, *args[1]
                        ),
                        self._retrieve_eval_logic_input(
                            run_overviews,
                            skip_example_on_any_failure=skip_example_on_any_failure,
                            num_examples=num_examples,
                        ),
                    ),
                    desc="Evaluating",
                )
            )

        failed_evaluation_count = sum(
            isinstance(example_evaluation, FailedExampleEvaluation)
            for example_evaluation in example_evaluations
        )

        successful_evaluation_count = len(example_evaluations) - failed_evaluation_count
        full_description = (
            self.description + " : " + description if description else self.description
        )
        overview = EvaluationOverview(
            run_overviews=frozenset(run_overviews),
            id=eval_id,
            start_date=start,
            end_date=utc_now(),
            successful_evaluation_count=successful_evaluation_count,
            failed_evaluation_count=failed_evaluation_count,
            description=full_description,
            labels=labels,
            metadata=metadata,
        )
        self._evaluation_repository.store_evaluation_overview(overview)

        return overview

    @final
    def evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        evaluation_id: str,
        abort_on_error: bool,
        *example_outputs: SuccessfulExampleOutput[Output],
    ) -> Evaluation | FailedExampleEvaluation:
        try:
            result: Evaluation | FailedExampleEvaluation = (
                self._evaluation_logic.do_evaluate(
                    example,
                    *example_outputs,
                )
            )
        except Exception as e:
            if abort_on_error:
                raise e
            print(
                f'FAILED EVALUATION: example "{example.id}", {type(e).__qualname__}: "{e}"'
            )
            result = FailedExampleEvaluation.from_exception(e)
        self._evaluation_repository.store_example_evaluation(
            ExampleEvaluation(
                evaluation_id=evaluation_id, example_id=example.id, result=result
            )
        )

        return result
