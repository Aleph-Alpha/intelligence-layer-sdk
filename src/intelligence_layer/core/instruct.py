from typing import Optional
from pydantic import BaseModel
from intelligence_layer.core.model import AlephAlphaModel, CompleteInput, CompleteOutput

from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan


class InstructInput(BaseModel):
    instruction: str
    input: Optional[str] = None
    response_prefix: Optional[str] = None
    maximum_tokens: int = 128


class Instruct(Task[InstructInput, CompleteOutput]):
    def __init__(self, model: AlephAlphaModel) -> None:
        super().__init__()
        self._model = model

    def do_run(self, input: InstructInput, task_span: TaskSpan) -> CompleteOutput:
        prompt = self._model.to_instruct_prompt(
            instruction=input.instruction,
            input=input.input,
            response_prefix=input.response_prefix
        )
        return self._model.complete(CompleteInput(
            prompt=prompt,
            maximum_tokens=input.maximum_tokens
        ), task_span)
