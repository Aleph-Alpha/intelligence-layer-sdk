from intelligence_layer.single_chunk_qa import (
    SingleChunkQaInput,
    QaOutput,
    SingleChunkQa,
    TextRange,
)
from intelligence_layer.task import DebugLog, LogLevel, Task
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    ExplanationRequest,
    ExplanationResponse,
    Prompt,
    TextScore,
)
from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from typing import List, Tuple
from pydantic import BaseModel


class MultipleChunkQaInput(BaseModel):
    chunks: List[str]
    question: str


class MultipleChunkQa(Task[MultipleChunkQaInput, QaOutput]):
    PROMPT_TEMPLATE = """### Instruction:
You will be given a number of Answers to a Question. Based on them, generate a single final answer.
Condense multiple answers into a single answer. Rely only on the provided answers. Don't use the world's knowledge. The answer should combine the individual answers. If the answers contradict each other, e.g., one saying that the colour is green and the other saying that the colour is black, say that there are contradicting answers saying the colour is green or the colour is black.

### Input:
Question: {{question}}

Answers:
{{answers}}

### Response:
Final answer:"""

    def __init__(
        self,
        client: Client,
        log_level: LogLevel,
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.log_level = log_level
        self.completion = Completion(client, log_level)
        self.single_chunk_qa = SingleChunkQa(client, log_level, model)

    def _prompt_text(
        self, text: str, question: str, no_answer_text: str
    ) -> Tuple[str, TextRange]:
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, no_answer_text=no_answer_text
        )
        # need to calculate the range differently
        return prompt, TextRange(start=len(prompt), end=len(prompt) + len(text))

    def run(self, input: MultipleChunkQaInput) -> QaOutput:
        """Executes the process for this use-case."""

        qa_outputs: List[QaOutput] = [
            self.single_chunk_qa.run(
                SingleChunkQaInput(question=input.question, chunk=chunk)
            )
            for chunk in input.chunks
        ]

        answers: List[str | None] = [output.answer for output in qa_outputs]

        return QaOutput(
            answer="XXX",
            highlights=["xxxx", "zzzz"],
            debug_log=DebugLog.enabled(level=self.log_level),
        )
