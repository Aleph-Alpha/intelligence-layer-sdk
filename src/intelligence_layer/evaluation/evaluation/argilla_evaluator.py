import random
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import combinations
from typing import Mapping, Optional, Sequence, cast

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core import CompleteOutput, Input, InstructInput, Output
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.argilla_evaluation_repository import (
    ArgillaEvaluationRepository,
    RecordDataSequence,
)
from intelligence_layer.evaluation.evaluation.async_evaluation import (
    AsyncEvaluationLogic,
    AsyncEvaluator,
)
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput
from intelligence_layer.evaluation.run.run_repository import RunRepository


class ArgillaEvaluationLogic(
    AsyncEvaluationLogic[Input, Output, ExpectedOutput, Evaluation], ABC
):
    def __init__(self, client: ArgillaClient):
        self._client = client

    def fields(self) -> Sequence[Field]:
        return [Field(name="name", title="title")]

    def questions(self) -> Sequence[Question]:
        return [
            Question(name="name", title="title", description="description", options=[0])
        ]

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

    def _from_record(argilla_evaluation: ArgillaEvaluation) -> Evaluation: ...


class ArgillaEvaluator(
    AsyncEvaluator[Input, Output, ExpectedOutput, ArgillaEvaluation]
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
        argilla_client: ArgillaClient,
        workspace_id: str,
    ) -> None:
        super().__init__(
            dataset_repository,
            run_repository,
            evaluation_repository,
            description,
            evaluation_logic,  # type: ignore
        )
        self._client = argilla_client
        self._workspace_id = workspace_id
        self._evaluation_logic: ArgillaEvaluationLogic[Input, Output, ExpectedOutput]

    def retrieve(
        self,
        evaluation_id: str,
    ) -> EvaluationOverview:
        example_evaluations = [
            ExampleEvaluation(
                evaluation_id=evaluation_id,
                example_id=example_evaluation.example_id,
                # cast to Evaluation because mypy thinks ArgillaEvaluation cannot be Evaluation
                result=self._from_record(example_evaluation),
            )
            for example_evaluation in self._client.evaluations(evaluation_id)
        ]
        evaluations = sorted(example_evaluations, key=lambda i: i.example_id)
        for evaluation in evaluations:
            self._evaluation_repository.store_example_evaluation(evaluation)

        return EvaluationOverview(
            run_overviews=frozenset(),
            id=id,
            start_date=datetime.now(),
            description="",
            end_date=datetime.now(),
            successful_evaluation_count=1,
            failed_evaluation_count=0,
            skipped_evaluation_count=0,
        )

    def evaluation_type(self) -> type[ArgillaEvaluation]:  # type: ignore
        return ArgillaEvaluation

    def submit(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        abort_on_error: bool = False,
    ) -> PartialEvaluationOverview:
        argilla_dataset_id = self._client.ensure_dataset_exists(
            self._workspace_id,
            dataset_name="name",
            fields=self._evaluation_logic.fields(),
            questions=self._evaluation_logic.questions(),
        )

        run_overviews = self._load_run_overviews(*run_ids)
        for example, outputs in self.retrieve_eval_logic_input(
            run_overviews, num_examples=num_examples
        ):
            self._evaluation_logic._to_record(example, outputs)
            record_sequence = self._evaluation_logic.submit(example, outputs)
            for record in record_sequence:
                self._client.add_record(self._workspace_id, record)

        return PartialEvaluationOverview(
            run_overviews=frozenset(run_overviews),
            id=argilla_dataset_id,
            start_date=datetime.now(),
            description=self.description,
        )


## An eval ids rankommen
# wo ergibt es sinn field und questions zu setzen?


class InstructComparisonArgillaEvaluationLogic(
    ArgillaEvaluationLogic[InstructInput, CompleteOutput, None]
):
    def __init__(
        self,
        workspace_id: str,
        fields: Mapping[str, Field],
        high_priority_runs: Optional[frozenset[str]] = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._fields = fields
        self._high_priority_runs = high_priority_runs

    def _to_record(
        self,
        example: Example[InstructInput, None],
        *outputs: SuccessfulExampleOutput[CompleteOutput],
    ) -> RecordDataSequence:
        pairs = combinations(outputs, 2)
        return RecordDataSequence(
            records=[
                self._create_record_data(example, first, second)
                for [first, second] in pairs
                if self._high_priority_runs is None
                or any(
                    run_id in self._high_priority_runs
                    for run_id in [first.run_id, second.run_id]
                )
            ]
        )

    def _create_record_data(
        self,
        example: Example[InstructInput, None],
        first: SuccessfulExampleOutput[CompleteOutput],
        second: SuccessfulExampleOutput[CompleteOutput],
    ) -> RecordData:
        if random.choice([True, False]):
            first, second = second, first
        return RecordData(
            content={
                self._fields["KEY_INSTRUCTION"].name: example.input.instruction,
                self._fields["KEY_INPUT"].name: example.input.input or "",
                self._fields["KEY_RESPONSE_1"].name: first.output.completion,
                self._fields["KEY_RESPONSE_2"].name: second.output.completion,
            },
            example_id=example.id,
            metadata={
                self._fields["KEY_RESPONSE_1"].name: first.run_id,
                self._fields["KEY_RESPONSE_2"].name: second.run_id,
            },
        )


def create_instruct_comparison_argilla_evaluation_classes(
    workspace_id: str,
    evaluation_repository: EvaluationRepository,
    argilla_client: ArgillaClient,
    high_priority_runs: Optional[frozenset[str]] = None,
) -> tuple[InstructComparisonArgillaEvaluationLogic, ArgillaEvaluationRepository]:
    KEY_INSTRUCTION = "instruction"
    KEY_INPUT = "input"
    KEY_RESPONSE_1 = "first"
    KEY_RESPONSE_2 = "second"
    KEY_QUESTION = "winner"
    OPTIONS = [1, 2, 3]

    fields = {
        "KEY_INSTRUCTION": Field(name=KEY_INSTRUCTION, title="Instruction"),
        "KEY_INPUT": Field(name=KEY_INPUT, title="Input"),
        "KEY_RESPONSE_1": Field(name=KEY_RESPONSE_1, title="Response 1"),
        "KEY_RESPONSE_2": Field(name=KEY_RESPONSE_2, title="Response 2"),
    }
    questions = [
        Question(
            name=KEY_QUESTION,
            title="Which response is better?",
            description="1: The first completion is better.\n2: The second completion is better.\n3: They are both equally good.",
            options=OPTIONS,
        )
    ]

    return InstructComparisonArgillaEvaluationLogic(
        workspace_id, fields, high_priority_runs
    ), ArgillaEvaluationRepository(
        evaluation_repository,
        argilla_client,
        workspace_id,
        list(fields.values()),
        questions,
    )
