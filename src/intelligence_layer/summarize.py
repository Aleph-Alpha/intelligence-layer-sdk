from aleph_alpha_client import Client, CompletionRequest, Prompt
from pydantic import BaseModel

from .completion import Completion, CompletionInput, CompletionOutput
from .task import DebugLog, LogLevel, Task


class SummarizeInput(BaseModel):
    text: str


class SummarizeOutput(BaseModel):
    summary: str
    debug_log: DebugLog


class ShortBodySummarize(Task[SummarizeInput, SummarizeOutput]):
    """Summarization Method to summarize a text in a few sentences.

    This task just takes the text as input and provides a summary as well as a debug log as
    the output."""

    PROMPT_TEMPLATE: str = """### Instruction:
Summarize in just one or two sentences.

### Input:
{text}

### Response:"""
    MODEL: str = "luminous-supreme-control"
    client: Client

    def __init__(self, client: Client, log_level: LogLevel) -> None:
        super().__init__()
        self.client = client
        self.log_level = log_level
        self.completion_task = Completion(client, log_level)

    def run(self, input: SummarizeInput) -> SummarizeOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        formatted_prompt = self._format_prompt(text=input.text, debug_log=debug_log)
        completion = self._complete(prompt=formatted_prompt, debug_log=debug_log)
        summary = completion.completion()
        return SummarizeOutput(summary=summary, debug_log=debug_log)

    def _format_prompt(self, text: str, debug_log: DebugLog) -> Prompt:
        prompt_str = self.PROMPT_TEMPLATE.format(text=text)
        debug_log.info(
            "Formatted prompt string",
            {
                "template": self.PROMPT_TEMPLATE,
                "formatted": prompt_str,
            },
        )
        return Prompt.from_text(prompt_str)

    def _complete(self, prompt: Prompt, debug_log: DebugLog) -> CompletionOutput:
        request = CompletionRequest(
            prompt=prompt,
            maximum_tokens=128,
            log_probs=3,
        )
        response = self.completion_task.run(
            CompletionInput(request=request, model=self.MODEL)
        )
        debug_log.debug(
            "Completion Request/Response",
            response.debug_log,
        )
        return response
