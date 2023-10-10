from aleph_alpha_client import Client, CompletionRequest, Prompt, PromptTemplate
from pydantic import BaseModel

from .completion import Completion, CompletionInput, CompletionOutput
from .task import DebugLogger, Task


class SummarizeInput(BaseModel):
    text: str


class SummarizeOutput(BaseModel):
    summary: str


class ShortBodySummarize(Task[SummarizeInput, SummarizeOutput]):
    """Summarization Method to summarize a text in a few sentences.

    This task just takes the text as input and provides a summary as well as a debug log as
    the output."""

    PROMPT_TEMPLATE: str = """### Instruction:
Summarize in just one or two sentences.

### Input:
{{text}}

### Response:"""
    MODEL: str = "luminous-supreme-control"
    client: Client

    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client
        self.completion_task = Completion(client)

    def run(self, input: SummarizeInput, logger: DebugLogger) -> SummarizeOutput:
        prompt = self._format_prompt(text=input.text, logger=logger)
        completion = self._complete(
            prompt=prompt, logger=logger.child_logger("Generate Summary")
        )
        return SummarizeOutput(summary=completion.completion())

    def _format_prompt(self, text: str, logger: DebugLogger) -> Prompt:
        logger.log(
            "Prompt template/text", {"template": self.PROMPT_TEMPLATE, "text": text}
        )
        prompt = PromptTemplate(self.PROMPT_TEMPLATE).to_prompt(text=text)
        return prompt

    def _complete(self, prompt: Prompt, logger: DebugLogger) -> CompletionOutput:
        request = CompletionRequest(
            prompt=prompt,
            maximum_tokens=128,
            log_probs=3,
        )
        response = self.completion_task.run(
            CompletionInput(request=request, model=self.MODEL),
            logger,
        )
        return response
