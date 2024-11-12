import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import combinations
from typing import Any, ClassVar, Optional
from uuid import uuid4

import argilla as rg  # type: ignore
from pydantic import BaseModel

from intelligence_layer.connectors import (
    ArgillaClient,
    ArgillaEvaluation,
    RecordData,
)
from intelligence_layer.connectors.base.json_serializable import (
    SerializableDict,
)
from intelligence_layer.core import CompleteOutput, Input, InstructInput, Output
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    ExampleEvaluation,
    FailedExampleEvaluation,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluator.async_evaluator import (
    AsyncEvaluationRepository,
    AsyncEvaluator,
)
from intelligence_layer.evaluation.evaluation.evaluator.base_evaluator import (
    EvaluationLogicBase,
)
from intelligence_layer.evaluation.evaluation.evaluator.incremental_evaluator import (
    ComparisonEvaluation,
    MatchOutcome,
)
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput
from intelligence_layer.evaluation.run.run_repository import RunRepository


class RecordDataSequence(BaseModel):
    records: Sequence[RecordData]


class ArgillaEvaluationLogic(
    EvaluationLogicBase[Input, Output, ExpectedOutput, Evaluation], ABC
):
    def __init__(self, fields: Mapping[str, Any], questions: Sequence[Any]):
        self.fields = fields
        self.questions = questions

    @abstractmethod
    def to_record(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> RecordDataSequence:
        """This method is responsible for translating the `Example` and `Output` of the task to :class:`RecordData`.

        The specific format depends on the `fields`.


        Args:
            example: The example to be translated.
            *output: The output of the example that was run.

        Returns:
            A :class:`RecordDataSequence` that contains entries that should be evaluated in Argilla.
        """
        ...

    @abstractmethod
    def from_record(self, argilla_evaluation: ArgillaEvaluation) -> Evaluation:
        """This method takes the specific Argilla evaluation format and converts into a compatible :class:`Evaluation`.

        The format of argilla_evaluation.responses depends on the `questions` attribute.
        Each `name` of a question will be a key in the `argilla_evaluation.responses` mapping.

        Args:
            argilla_evaluation: Argilla-specific data for a single evaluation.

        Returns:
            An :class:`Evaluation` that contains all evaluation specific data.
        """


class ArgillaEvaluator(AsyncEvaluator[Input, Output, ExpectedOutput, Evaluation]):
    """Evaluator used to integrate with Argilla (https://github.com/argilla-io/argilla).

    Use this evaluator if you would like to easily do human eval.
    This evaluator runs a dataset and sends the input and output to Argilla to be evaluated.

    Arguments:
        dataset_repository: The repository with the examples that will be taken for the evaluation.
        run_repository: The repository of the runs to evaluate.
        evaluation_repository: The repository that will be used to store evaluation results.
        description: Human-readable description for the evaluator.
        evaluation_logic: The logic to use for evaluation.
        argilla_client: The client to interface with argilla.
        workspace_id: The argilla workspace id where datasets are created for evaluation.

    See the :class:`EvaluatorBase` for more information.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        run_repository: RunRepository,
        evaluation_repository: AsyncEvaluationRepository,
        description: str,
        evaluation_logic: ArgillaEvaluationLogic[
            Input, Output, ExpectedOutput, Evaluation
        ],
        argilla_client: ArgillaClient,
        workspace_id: str,
    ) -> None:
        super().__init__(
            dataset_repository,
            run_repository,
            evaluation_repository,
            description,
            evaluation_logic,
        )
        self._client = argilla_client
        self._workspace_id = workspace_id
        self._evaluation_logic: ArgillaEvaluationLogic[
            Input, Output, ExpectedOutput, Evaluation
        ]
        self._evaluation_repository: AsyncEvaluationRepository

    def submit(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
        dataset_name: Optional[str] = None,
        abort_on_error: bool = False,
        skip_example_on_any_failure: bool = True,
        labels: Optional[set[str]] = None,
        metadata: Optional[SerializableDict] = None,
    ) -> PartialEvaluationOverview:
        if metadata is None:
            metadata = dict()
        if labels is None:
            labels = set()
        argilla_dataset_id = self._client.create_dataset(
            self._workspace_id,
            dataset_name if dataset_name else str(uuid4()),
            fields=list(self._evaluation_logic.fields.values()),
            questions=self._evaluation_logic.questions,
        )

        run_overviews = self._load_run_overviews(*run_ids)
        submit_count = 0
        for example, outputs in self._retrieve_eval_logic_input(
            run_overviews,
            skip_example_on_any_failure=skip_example_on_any_failure,
            num_examples=num_examples,
        ):
            record_sequence = self._evaluation_logic.to_record(example, *outputs)
            for record in record_sequence.records:
                try:
                    self._client.add_record(argilla_dataset_id, record)
                    submit_count += 1
                except Exception as e:
                    if abort_on_error:
                        raise e
                    evaluation = FailedExampleEvaluation.from_exception(e)
                    self._evaluation_repository.store_example_evaluation(
                        ExampleEvaluation(
                            evaluation_id=argilla_dataset_id,
                            example_id=example.id,
                            result=evaluation,
                        )
                    )
                    print(
                        f"Uploading a record to argilla failed with the following error:\n{e}"
                    )

        partial_overview = PartialEvaluationOverview(
            run_overviews=frozenset(run_overviews),
            id=argilla_dataset_id,
            start_date=datetime.now(),
            submitted_evaluation_count=submit_count,
            description=self.description,
            labels=labels,
            metadata=metadata,
        )

        self._evaluation_repository.store_partial_evaluation_overview(partial_overview)
        return partial_overview

    def retrieve(
        self,
        partial_evaluation_id: str,
    ) -> EvaluationOverview:
        partial_overview = self._evaluation_repository.partial_evaluation_overview(
            partial_evaluation_id
        )
        if not partial_overview:
            raise ValueError(
                f"Partial overview for evaluation id {partial_evaluation_id} not found."
            )

        example_evaluations = [
            ExampleEvaluation(
                evaluation_id=partial_evaluation_id,
                example_id=example_evaluation.example_id,
                # cast to Evaluation because mypy thinks ArgillaEvaluation cannot be Evaluation
                result=self._evaluation_logic.from_record(example_evaluation),
            )
            for example_evaluation in self._client.evaluations(partial_evaluation_id)
        ]
        evaluations = sorted(example_evaluations, key=lambda i: i.example_id)

        for evaluation in evaluations:
            self._evaluation_repository.store_example_evaluation(evaluation)
        num_failed_evaluations = len(
            self._evaluation_repository.failed_example_evaluations(
                partial_evaluation_id, self.evaluation_type()
            )
        )
        num_not_yet_evaluated_evals = partial_overview.submitted_evaluation_count - len(
            evaluations
        )

        overview = EvaluationOverview(
            run_overviews=partial_overview.run_overviews,
            id=partial_evaluation_id,
            start_date=partial_overview.start_date,
            description=partial_overview.description,
            end_date=datetime.now(),
            successful_evaluation_count=len(evaluations),
            failed_evaluation_count=num_not_yet_evaluated_evals
            + num_failed_evaluations,
            labels=partial_overview.labels,
            metadata=partial_overview.metadata,
        )
        self._evaluation_repository.store_evaluation_overview(overview)
        return overview


class InstructComparisonArgillaEvaluationLogic(
    ArgillaEvaluationLogic[InstructInput, CompleteOutput, None, ComparisonEvaluation]
):
    KEY_INSTRUCTION = "instruction"
    KEY_INPUT = "input"
    KEY_RESPONSE_1 = "first"
    KEY_RESPONSE_2 = "second"
    KEY_QUESTION = "winner"
    OPTIONS: ClassVar[list[int]] = [1, 2, 3]

    def __init__(
        self,
        high_priority_runs: Optional[frozenset[str]] = None,
    ) -> None:
        self._high_priority_runs = high_priority_runs
        super().__init__(
            fields={
                "KEY_INSTRUCTION": rg.TextField(
                    name=self.KEY_INSTRUCTION, title="Instruction"
                ),
                "KEY_INPUT": rg.TextField(name=self.KEY_INPUT, title="Input"),
                "KEY_RESPONSE_1": rg.TextField(
                    name=self.KEY_RESPONSE_1, title="Response 1"
                ),
                "KEY_RESPONSE_2": rg.TextField(
                    name=self.KEY_RESPONSE_2, title="Response 2"
                ),
            },
            questions=[
                rg.RatingQuestion(
                    name=self.KEY_QUESTION,
                    title="Which response is better?",
                    description="1: The first completion is better.\n2: The second completion is better.\n3: They are both equally good.",
                    values=self.OPTIONS,
                )
            ],
        )

    def to_record(
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
                self.fields["KEY_INSTRUCTION"].name: example.input.instruction,
                self.fields["KEY_INPUT"].name: example.input.input or "",
                self.fields["KEY_RESPONSE_1"].name: first.output.completion,
                self.fields["KEY_RESPONSE_2"].name: second.output.completion,
            },
            example_id=example.id,
            metadata={
                self.fields["KEY_RESPONSE_1"].name: first.run_id,
                self.fields["KEY_RESPONSE_2"].name: second.run_id,
            },
        )

    def from_record(
        self, argilla_evaluation: ArgillaEvaluation
    ) -> ComparisonEvaluation:
        return ComparisonEvaluation(
            first_player=argilla_evaluation.metadata[
                self.fields["KEY_RESPONSE_1"].name
            ],
            second_player=argilla_evaluation.metadata[
                self.fields["KEY_RESPONSE_2"].name
            ],
            outcome=MatchOutcome.from_rank_literal(
                int(argilla_evaluation.responses["winner"])
            ),
        )
