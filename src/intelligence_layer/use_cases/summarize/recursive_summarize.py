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
        n_loops: The number of times to recursively summarize.
    """

    text: str
    language: Language = Language("en")
    n_loops: int = 2


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
        for n in range(input.n_loops):
            summarize_output = self.long_context_summarize_task.run(
                LongContextSummarizeInput(text=text, language=input.language), task_span
            )
            text = "\n".join(
                partial_summary.summary
                for partial_summary in summarize_output.partial_summaries
            )

            if len(summarize_output.partial_summaries) == 1:
                task_span.log(
                    message="Stopped recursion.", value=f"condensed {n}-times"
                )
                break

        return SummarizeOutput(summary=text)
