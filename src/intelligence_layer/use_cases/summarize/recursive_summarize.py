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
    text: str
    language: Language = Language("en")
    n_loops: int = 3


class RecursiveSummarize(Task[RecursiveSummarizeInput, SummarizeOutput]):
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
        for _ in range(input.n_loops):
            summarize_output = self.long_context_summarize_task.run(
                LongContextSummarizeInput(text=text, language=input.language), task_span
            )
            text = "\n".join(
                partial_summary.summary
                for partial_summary in summarize_output.partial_summaries
            )

        return SummarizeOutput(summary=text)
