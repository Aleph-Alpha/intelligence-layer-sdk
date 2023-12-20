from typing import Iterable, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors import Field
from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaEvaluation,
    Question,
    RecordData,
)
from intelligence_layer.core.complete import InstructInput, PromptOutput
from intelligence_layer.core.evaluation.domain import Example, SuccessfulExampleOutput
from intelligence_layer.core.evaluation.evaluator import (
    ArgillaEvaluationRepository,
    ArgillaEvaluator,
    DatasetRepository,
)


class EloScore(BaseModel):
    pass


class EloScoreArgillaEvaluator(
    ArgillaEvaluator[
        InstructInput,
        PromptOutput,
        None,
        EloScore,
    ]
):
    def __init__(
        self,
        evaluation_repository: ArgillaEvaluationRepository,
        dataset_repository: DatasetRepository,
        workspace_id: str,
    ) -> None:
        fields = [
            Field(name="instruction", title="Instruction"),
            Field(name="input", title="Input"),
            Field(name="response1", title="Response1"),
            Field(name="response2", title="Response2"),
        ]
        questions = [
            Question(
                name="winner",
                title="which response is better",
                description="3 means they are both equally good",
                options=[1, 2, 3],
            )
        ]

        super().__init__(
            evaluation_repository,
            dataset_repository,
            workspace_id,
            fields,
            questions,
        )

    def _to_record(
        self,
        example: Example[InstructInput, None],
        *example_outputs: SuccessfulExampleOutput[PromptOutput]
    ) -> Sequence[RecordData]:
        return [
            RecordData(
                content={
                    "instruction": example.input.instruction,
                    "input": example.input.input or "",
                    "response1": example_outputs[0].output.completion,
                    "response2": example_outputs[1].output.completion,
                },
                example_id=example.id,
            )
        ]

    def aggregate(self, evaluations: Iterable[ArgillaEvaluation]) -> EloScore:
        return EloScore()
