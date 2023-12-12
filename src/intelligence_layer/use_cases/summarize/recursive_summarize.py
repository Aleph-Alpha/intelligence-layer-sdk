from typing import Optional

from pydantic import BaseModel

from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan
from intelligence_layer.use_cases.summarize.summarize import (
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    SummarizeOutput,
)


class RecursiveSummarizeInput(BaseModel):
    """The Input for a recursive summarize task.

    Attributes:
        text: A text of any length.
        language: The desired language of the summary. ISO 619 str with language e.g. en, fr, etc.
        max_tokens: The max number of tokens to be in the final summary.
        max_loops: The max number of times to recursively summarize.
    """

    text: str
    language: Language = Language("en")
    max_tokens: Optional[int] = None
    max_loops: Optional[int] = None


class RecursiveSummarize(Task[RecursiveSummarizeInput, SummarizeOutput]):
    """Condenses a text recursively by summarizing summaries.

    Uses any long-context summarize task to go over text recursively and condense it even further.

    Args:
        long_context_summarize_task: Any task that satifies the interface Input: LongContextSummarizeInput and Output: LongContextSummarizeOutput.
    """

    def __init__(
        self,
        long_context_summarize_task: Task[
            LongContextSummarizeInput, LongContextSummarizeOutput
        ],
    ) -> None:
        self.long_context_summarize_task = long_context_summarize_task

    def do_run(
        self, input: RecursiveSummarizeInput, task_span: TaskSpan
    ) -> SummarizeOutput:
        text = input.text
        continue_loop = True
        loop_count = 0
        while continue_loop:
            summarize_output = self.long_context_summarize_task.run(
                LongContextSummarizeInput(text=text, language=input.language), task_span
            )
            num_generated_tokens = 0
            text = ""
            for partial_summary in summarize_output.partial_summaries:
                num_generated_tokens += partial_summary.generated_tokens
                text += partial_summary.summary + "\n"

            loop_count += 1

            if len(summarize_output.partial_summaries) == 1:
                continue_loop = False

            elif input.max_tokens and num_generated_tokens < input.max_tokens:
                continue_loop = False

            elif input.max_loops and loop_count <= input.max_loops:
                continue_loop = False

        return SummarizeOutput(
            summary=text.strip(), generated_tokens=num_generated_tokens
        )
