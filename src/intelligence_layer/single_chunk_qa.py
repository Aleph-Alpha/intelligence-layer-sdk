from typing import Optional, Sequence
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    Prompt,
)
from pydantic import BaseModel

from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from intelligence_layer.text_highlight import (
    TextHighlight,
    TextHighlightInput,
)
from intelligence_layer.prompt_template import (
    PromptTemplate,
    PromptWithMetadata,
)
from intelligence_layer.task import (
    DebugLogger,
    Evaluation,
    Evaluator,
    Task,
)


class SingleChunkQaInput(BaseModel):
    chunk: str
    question: str


class SingleChunkQaOutput(BaseModel):
    answer: Optional[str]
    highlights: Sequence[str]


class SingleChunkQa(Task[SingleChunkQaInput, SingleChunkQaOutput]):
    PROMPT_TEMPLATE_STR = """### Instruction:
{{question}}
If there's no answer, say "{{no_answer_text}}".

### Input:
{% promptrange text %}{{text}}{% endpromptrange %}

### Response:"""
    NO_ANSWER_STR = "NO_ANSWER_IN_TEXT"

    def __init__(
        self,
        client: Client,
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.completion = Completion(client)
        self.text_highlight = TextHighlight(client)
        self.model = model

    def run(
        self, input: SingleChunkQaInput, logger: DebugLogger
    ) -> SingleChunkQaOutput:
        prompt_with_metadata = self._to_prompt_with_metadata(
            input.chunk, input.question
        )
        output = self._complete(
            prompt_with_metadata.prompt, logger.child_logger("Generate Answer")
        )
        highlights = self._get_highlights(
            prompt_with_metadata,
            output.completion(),
            logger.child_logger("Explain Answer"),
        )
        return SingleChunkQaOutput(
            answer=self._no_answer_to_none(output.completion().strip()),
            highlights=highlights,
        )

    def _to_prompt_with_metadata(self, text: str, question: str) -> PromptWithMetadata:
        template = PromptTemplate(self.PROMPT_TEMPLATE_STR)
        return template.to_prompt_with_metadata(
            text=text, question=question, no_answer_text=self.NO_ANSWER_STR
        )

    def _complete(self, prompt: Prompt, logger: DebugLogger) -> CompletionOutput:
        request = CompletionRequest(prompt)
        output = self.completion.run(
            CompletionInput(request=request, model=self.model), logger
        )
        return output

    def _get_highlights(
        self,
        prompt_with_metadata: PromptWithMetadata,
        completion: str,
        logger: DebugLogger,
    ) -> Sequence[str]:
        highlight_input = TextHighlightInput(
            prompt_with_metadata=prompt_with_metadata,
            target=completion,
            model=self.model,
        )
        highlight_output = self.text_highlight.run(highlight_input, logger)
        return [h.text for h in highlight_output.highlights if h.score > 0]

    def _no_answer_to_none(self, completion: str) -> Optional[str]:
        return completion if completion != self.NO_ANSWER_STR else None
