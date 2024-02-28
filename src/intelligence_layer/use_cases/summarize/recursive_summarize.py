from intelligence_layer.core import Task, TaskSpan
from intelligence_layer.use_cases.summarize.summarize import (
    LongContextSummarizeInput,
    LongContextSummarizeOutput,
    SummarizeOutput,
)


class RecursiveSummarize(Task[LongContextSummarizeInput, SummarizeOutput]):
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
        self, input: LongContextSummarizeInput, task_span: TaskSpan
    ) -> SummarizeOutput:
        num_partial_summaries = 0
        text = input.text
        while True:
            summarize_output = self.long_context_summarize_task.run(
                LongContextSummarizeInput(text=text, language=input.language), task_span
            )
            if num_partial_summaries == len(summarize_output.partial_summaries):
                break
            num_partial_summaries = len(summarize_output.partial_summaries)

            num_generated_tokens = 0
            text = ""
            for partial_summary in summarize_output.partial_summaries:
                num_generated_tokens += partial_summary.generated_tokens
                text += partial_summary.summary + "\n"

            if len(summarize_output.partial_summaries) == 1:
                break

            elif input.max_tokens and num_generated_tokens < input.max_tokens:
                break

        return SummarizeOutput(
            summary=text.strip(), generated_tokens=num_generated_tokens
        )
