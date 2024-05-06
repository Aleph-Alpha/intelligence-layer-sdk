import random
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import combinations
from typing import Generic, Iterable, Mapping, Optional, Sequence, Tuple, cast
import typing
from uuid import uuid4

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Field,
    Question,
    RecordData,
)
from intelligence_layer.core import CompleteOutput, Input, InstructInput, Output
from intelligence_layer.core.tracer.tracer import utc_now
from intelligence_layer.evaluation.dataset.dataset_repository import DatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.argilla_evaluation_repository import (
    ArgillaEvaluationRepository,
    RecordDataSequence,
)
from intelligence_layer.evaluation.evaluation.async_evaluation import AsyncEvaluator
from intelligence_layer.evaluation.evaluation.domain import (
    Evaluation,
    EvaluationOverview,
    PartialEvaluationOverview,
)
from intelligence_layer.evaluation.evaluation.evaluation_repository import (
    EvaluationRepository,
)
from intelligence_layer.evaluation.evaluation.evaluator import (
    EvaluationLogic,
    Evaluator,
)
from intelligence_layer.evaluation.run.domain import (
    ExampleOutput,
    FailedExampleRun,
    RunOverview,
    SuccessfulExampleOutput,
)
from intelligence_layer.evaluation.run.run_repository import RunRepository


class ArgillaEvaluationLogic(
    EvaluationLogic[Input, Output, ExpectedOutput, RecordDataSequence], ABC
):
    def fields(self) -> Sequence[Field]:
        return [Field(name="name", title="title")]

    def questions(self) -> Sequence[Question]:
        return [
            Question(name="name", title="title", description="description", options=[0])
        ]

    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> RecordDataSequence:
        # Hier eher download logic als to-record
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

    def retrieve(
        self,
        argilla_id: str,
    ) -> EvaluationOverview:
        return EvaluationOverview(
            run_overviews=frozenset(),
            id=argilla_id,
            start_date=datetime.now(),
            description="",
            end_date=datetime.now(),
            successful_example_count=1,
            failed_example_count=0,
        )

    def evaluation_type(self) -> type[ArgillaEvaluation]:  # type: ignore
        return ArgillaEvaluation

    # Submission logic vom evaluate in submit (ArgillaEvaluator methods)

    def load_run_overviews(self, *run_ids: str) -> Sequence[RunOverview]:
        if not run_ids:
            raise ValueError("At least one run-id needs to be provided")
        run_overviews = set()
        for run_id in run_ids:
            run_overview = self._run_repository.run_overview(run_id)
            if not run_overview:
                raise ValueError(f"No RunOverview found for run-id: {run_id}")
            run_overviews.add(run_overview)
        return run_overviews

    def raise_if_overviews_have_different_dataset(*run_overviews: RunOverview):
        if not all(
            next(iter(run_overviews)).dataset_id == run_overview.dataset_id
            for run_overview in run_overviews
        ):
            raise ValueError(
                f"All run-overviews must reference the same dataset: {run_overviews}"
            )

    def retrieve_example_outputs(
        self, run_overviews: Sequence[RunOverview]
    ) -> Iterable[tuple[ExampleOutput[Output], ...]]:
        # this uses the assumption that the example outputs are sorted
        example_outputs_for_example: Iterable[tuple[ExampleOutput[Output], ...]] = zip(
            *(
                self._run_repository.example_outputs(
                    run_overview.id, self.output_type()
                )
                for run_overview in run_overviews
            ),
            strict=True,
        )

        return example_outputs_for_example

    def retrieve_examples(self, dataset_id: str):
        examples = self._dataset_repository.examples(
            dataset_id,
            self.input_type(),
            self.expected_output_type(),
        )
        if examples is None:
            raise ValueError(f"Dataset: {dataset_id} not found")

    def generate_evaluation_inputs(
        self, examples, example_outputs_for_example
    ):
        for example, example_outputs in zip(examples, example_outputs_for_example):
            successful_example_outputs = [
                cast(SuccessfulExampleOutput[Output], output)
                for output in example_outputs
                if not isinstance(output.output, FailedExampleRun)
            ]

            if not successful_example_outputs:
                continue

            yield (
                example,
                successful_example_outputs,
            )
            # if num_examples and current_example >= num_examples:
            #     break
            # current_example += 1

    def retrieve_eval_logic_input(
        self,
        *run_ids: str,
        num_examples: Optional[int] = None,
    ) -> Iterable[
        Tuple[
            Example[Input, ExpectedOutput],
            str,
            Sequence[SuccessfulExampleOutput[Output]],
        ]
    ]:
        start = utc_now()
        run_overviews = self.load_run_overviews(run_ids)
        self.raise_if_overviews_have_different_dataset(run_overviews)
        eval_id = self._evaluation_repository.initialize_evaluation()
        example_outputs_for_example = self.retrieve_example_outputs(run_overviews)
        dataset_id = next(iter(run_overviews)).dataset_id
        examples = self.retrieve_examples(dataset_id)
        return self.generate_evaluation_inputs(
            examples, example_outputs_for_example, eval_id
        )

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

        record_sequence = self._evaluation_logic._to_record()

        # if isinstance(record_sequence, RecordDataSequence):
        #     for record in record_sequence:
        #         self._client.add_record(workspace_id, record)
        #         pass
        # else:
        #     raise TypeError(
        #         "Argilla does not support submitting non-RecordDataSequence."
        #     )

        return PartialEvaluationOverview(
            run_overviews=[],
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

    def _do_submit(self) -> None:
        # Hier
        return


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
