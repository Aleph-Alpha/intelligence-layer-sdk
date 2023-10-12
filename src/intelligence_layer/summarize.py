from typing import Sequence
from aleph_alpha_client import Client, CompletionRequest, Prompt
from pydantic import BaseModel

from .completion import Completion, CompletionInput, CompletionOutput
from .prompt_template import PromptTemplate, PromptWithMetadata
from .text_highlight import TextHighlight, TextHighlightInput
from .task import DebugLogger, Task


class SummarizeInput(BaseModel):
    text: str


class SummarizeOutput(BaseModel):
    summary: str
    highlights: Sequence[str]


class ShortBodySummarize(Task[SummarizeInput, SummarizeOutput]):
    """Summarization Method to summarize a text in a few sentences.

    This task just takes the text as input and provides a summary as well as a debug log as
    the output."""

    PROMPT_TEMPLATE: str = """### Instruction:
Summarize in just one or two sentences.

### Input:
{% promptrange text %}{{text}}{% endpromptrange %}

### Response:"""
    MODEL: str = "luminous-supreme-control"
    client: Client

    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client
        self.completion = Completion(client)
        self.text_highlight = TextHighlight(client)

    def run(self, input: SummarizeInput, logger: DebugLogger) -> SummarizeOutput:
        prompt_with_metadata = self._format_prompt(text=input.text, logger=logger)
        completion = self._complete(
            prompt_with_metadata.prompt, logger.child_logger("Generate Summary")
        )
        highlights = self._get_highlights(
            prompt_with_metadata, completion.completion(), logger
        )
        return SummarizeOutput(summary=completion.completion(), highlights=highlights)

    def _format_prompt(self, text: str, logger: DebugLogger) -> PromptWithMetadata:
        logger.log(
            "Prompt template/text", {"template": self.PROMPT_TEMPLATE, "text": text}
        )
        prompt_with_metadata = PromptTemplate(
            self.PROMPT_TEMPLATE
        ).to_prompt_with_metadata(text=text)
        return prompt_with_metadata

    def _complete(self, prompt: Prompt, logger: DebugLogger) -> CompletionOutput:
        request = CompletionRequest(
            prompt=prompt,
            maximum_tokens=128,
            log_probs=3,
        )
        response = self.completion.run(
            CompletionInput(request=request, model=self.MODEL),
            logger,
        )
        return response

    def _get_highlights(
        self,
        prompt_with_metadata: PromptWithMetadata,
        completion: str,
        logger: DebugLogger,
    ) -> Sequence[str]:
        highlight_input = TextHighlightInput(
            prompt_with_metadata=prompt_with_metadata,
            target=completion,
            model=self.MODEL,
        )
        highlight_output = self.text_highlight.run(highlight_input, logger)
        return [h.text for h in highlight_output.highlights if h.score > 0]
