from typing import Optional
from aleph_alpha_client import Client, CompletionRequest, PromptTemplate
from pydantic import BaseModel
from intelligence_layer.completion import Completion, CompletionInput

from intelligence_layer.task import DebugLog, LogLevel, Task


class QaInput(BaseModel):
    text: str
    question: str


class QaOutput(BaseModel):
    answer: Optional[str]
    debug_log: DebugLog


NO_ANSWER_TEXT = "NO_ANSWER_IN_TEXT"


class Qa(Task[QaInput, QaOutput]):
    TEMPLATE_STR = """### Instruction:
{{question}} If there's no answer, say "{{no_answer_text}}".

### Input:
{{text}}

### Response:
"""
    MODEL = "luminous-extended-control"

    def __init__(self, client: Client, log_level: LogLevel):
        self.client = client
        self.log_level = log_level
        self.completion = Completion(client, log_level)

    def run(self, input: QaInput) -> QaOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        prompt = PromptTemplate(self.TEMPLATE_STR).to_prompt(
            question=input.question, text=input.text, no_answer_text=NO_ANSWER_TEXT
        )
        request = CompletionRequest(prompt)
        output = self.completion.run(CompletionInput(request=request, model=self.MODEL))
        debug_log.debug("Completion", output.debug_log)
        completion = output.response.completions[0].completion
        return QaOutput(
            answer=completion if completion != NO_ANSWER_TEXT else None,
            debug_log=debug_log,
        )
