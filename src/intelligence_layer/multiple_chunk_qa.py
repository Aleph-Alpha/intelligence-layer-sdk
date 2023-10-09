from typing import Optional, Sequence, Tuple
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


class MutipleChunkQaOutput(BaseModel):
    final_answer: Optional[str]
    sources: Sequence[Optional[str]]
    sources_highlights: Sequence[Sequence[str]]
    debug_log: DebugLog


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

    def _prompt_text(self, question: str, answers: Sequence[str]) -> str:
        prompt = self.PROMPT_TEMPLATE.format(question=question, answers=answers)
        return prompt

    def _complete(self, prompt: str, debug_log: DebugLog) -> CompletionOutput:
        request = CompletionRequest(Prompt.from_text(prompt))
        output = self.completion.run(CompletionInput(request=request, model=self.model))
        debug_log.debug("Completion", output.debug_log)
        debug_log.info(
            "Completion Input/Output",
            {"prompt": prompt, "completion": output.completion()},
        )
        return output

    def run(self, input: MultipleChunkQaInput) -> QaOutput:
        """Executes the process for this use-case."""

        qa_outputs: List[QaOutput] = [
            self.single_chunk_qa.run(
                SingleChunkQaInput(question=input.question, chunk=chunk)
            )
            for chunk in input.chunks
        ]

        answers: List[str | None] = [output.answer for output in qa_outputs]

        debug_log = DebugLog.enabled(level=self.log_level)
        prompt_text = self._prompt_text(
            input.question, [answer for answer in answers if answer is not None]
        )
        output = self._complete(prompt_text, debug_log)
        # TODO fix the prompt execution
        explanation = None
        return QaOutput(
            answer=self._no_answer_to_none(output.completion().strip()),
            highlights=self._to_highlights(explanation, input.chunk, [], debug_log),
            debug_log=debug_log,
        )
