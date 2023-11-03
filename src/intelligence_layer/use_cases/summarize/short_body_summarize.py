from typing import Sequence

from aleph_alpha_client import Client

from intelligence_layer.core.complete import (
    Instruct,
    InstructInput,
    InstructOutput,
)
from intelligence_layer.core.prompt_template import PromptWithMetadata
from intelligence_layer.core.task import Task
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.text_highlight import TextHighlight, TextHighlightInput
from intelligence_layer.use_cases.summarize.summarize import SummarizeInput, SummarizeOutput


class ShortBodySummarize(Task[SummarizeInput, SummarizeOutput]):
    """Summarises a section into a short text.

    Generate a short body natural language summary.
    Will also return highlights explaining which parts of the input contributed strongly to the completion.

    Note:
        `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        MAXIMUM_RESPONSE_TOKENS: The maximum number of tokens the summary will contain.
        INSTRUCTION: The verbal instruction sent to the model to make it generate the summary.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = ShortBodySummarize(client)
        >>> input = SummarizeInput(
        >>>     chunk="This is a story about pizza. Tina hates pizza. However, Mike likes it. Pete strongly believes that pizza is the best thing to exist."
        >>> )
        >>> logger = InMemoryLogger(name="Short Body Summarize")
        >>> output = task.run(input, logger)
        >>> print(output.summary)
        Tina does not like pizza, but Mike and Pete do.
    """

    MAXIMUM_RESPONSE_TOKENS = 128
    INSTRUCTION = "Summarize in just one or two sentences."
    _client: Client

    def __init__(self, client: Client, model: str = "luminous-supreme-control") -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._instruction = Instruct(client)
        self._text_highlight = TextHighlight(client)

    def run(self, input: SummarizeInput, logger: DebugLogger) -> SummarizeOutput:
        instruction_output = self._instruct(input.chunk, logger)
        highlights = self._get_highlights(
            instruction_output.prompt_with_metadata, instruction_output.response, logger
        )
        return SummarizeOutput(
            summary=instruction_output.response, highlights=highlights
        )

    def _instruct(self, input: str, logger: DebugLogger) -> InstructOutput:
        return self._instruction.run(
            InstructInput(
                instruction=self.INSTRUCTION,
                input=input,
                maximum_response_tokens=self.MAXIMUM_RESPONSE_TOKENS,
                model=self._model,
            ),
            logger,
        )

    def _get_highlights(
        self,
        prompt_with_metadata: PromptWithMetadata,
        completion: str,
        logger: DebugLogger,
    ) -> Sequence[str]:
        highlight_input = TextHighlightInput(
            prompt_with_metadata=prompt_with_metadata,
            target=completion,
            model=self._model,
            focus_ranges=frozenset({"input"}),
        )
        highlight_output = self._text_highlight.run(highlight_input, logger)
        return [h.text for h in highlight_output.highlights if h.score > 0]
