from pydantic import BaseModel

from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.core.detect_language import Language
from intelligence_layer.examples.summarize.steerable_long_context_summarize import (
    SteerableLongContextSummarize,
)
from intelligence_layer.examples.summarize.summarize import (
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    SummarizeOutput,
)


class RecursiveSummarizeInput(BaseModel):
    """The input for a recursive summarize-task for a text of any length.

    Attributes:
        text: A text of any length.
        language: The desired language of the summary. ISO 619 str with language e.g. en, fr, etc.
        max_tokens: The maximum desired length of the summary in tokens.
    """

    text: str
    language: Language = Language("en")
    max_tokens: int = 512


class RecursiveSummarize(Task[RecursiveSummarizeInput, SummarizeOutput]):
    """This task will summarize the input text recursively until the desired length is reached.
    It uses any long-context summarize task to go over text recursively and condense it even further.

        Args:
            long_context_summarize_task: Any task that satifies the interface Input: LongContextSummarizeInput and Output: LongContextSummarizeOutput.
                Defaults to :class:`SteerableLongContextSummarize`
    """

    def __init__(
        self,
        long_context_summarize_task: (
            Task[LongContextSummarizeInput, LongContextSummarizeOutput] | None
        ) = None,
    ) -> None:
        self.long_context_summarize_task = (
            long_context_summarize_task or SteerableLongContextSummarize()
        )

    def do_run(
        self, input: RecursiveSummarizeInput, task_span: TaskSpan
    ) -> SummarizeOutput:
        num_partial_summaries = 0
        text_to_summarize = input.text
        summary = ""
        num_generated_tokens = 0
        while True:
            summarize_output = self.long_context_summarize_task.run(
                LongContextSummarizeInput(
                    text=text_to_summarize, language=input.language
                ),
                task_span,
            )
            # If the number of chunks stayed the same, we assume that no further summarization has taken place and we return the previous summary
            if num_partial_summaries == len(summarize_output.partial_summaries):
                break
            num_partial_summaries = len(summarize_output.partial_summaries)

            partial_summaries = summarize_output.partial_summaries
            num_generated_tokens = sum(
                partial_summary.generated_tokens
                for partial_summary in partial_summaries
            )
            summary = "\n".join(
                partial_summary.summary for partial_summary in partial_summaries
            )
            # If the number of chunks is 1 we want to return the new summary since we assume that no further summarization will take place with our prompt
            if (
                len(summarize_output.partial_summaries) == 1
                or num_generated_tokens < input.max_tokens
            ):
                break
            text_to_summarize = summary

        return SummarizeOutput(
            summary=summary.strip(), generated_tokens=num_generated_tokens
        )
