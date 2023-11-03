from abc import abstractmethod
from typing import Sequence
from aleph_alpha_client import Client

from pydantic import BaseModel
from intelligence_layer.core.complete import PromptOutput
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.prompt_template import PromptWithMetadata

from intelligence_layer.core.task import Chunk, Task
from intelligence_layer.core.text_highlight import TextHighlight, TextHighlightInput


class SummarizeInput(BaseModel):
    """The input for a `Summarize` task.

    Attributes:
        chunk: The text chunk to be summarized.
    """

    chunk: Chunk
    use_highlights: bool


class SummarizeOutput(BaseModel):
    """The output of a `Summarize` task.

    Attributes:
        summary: The summary generated by the task.
        highlights: Highlights indicating which parts of the chunk contributed to the summary.
            Each highlight is a quote from the text.
    """

    summary: str
    highlights: Sequence[str]


class BaseSummarize(Task[SummarizeInput, SummarizeOutput]):
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

    _client: Client

    def __init__(self, client: Client, model: str = "luminous-supreme-control") -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._text_highlight = TextHighlight(client)

    def run(self, input: SummarizeInput, logger: DebugLogger) -> SummarizeOutput:
        prompt_output = self.get_prompt_and_complete(input, logger)
        highlights = (
            self._get_highlights(
                prompt_output.prompt_with_metadata, prompt_output.response, logger
            )
            if input.use_highlights
            else []
        )
        return SummarizeOutput(
            summary=prompt_output.response.strip(), highlights=highlights
        )

    @abstractmethod
    def get_prompt_and_complete(
        self, input: SummarizeInput, logger: DebugLogger
    ) -> PromptOutput:
        ...

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
