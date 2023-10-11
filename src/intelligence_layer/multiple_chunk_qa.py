from typing import Optional, Sequence 
from intelligence_layer.single_chunk_qa import (
    SingleChunkQaInput,
    SingleChunkQaOutput,
    SingleChunkQa,
)
from intelligence_layer.task import DebugLogger, Task
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    Prompt,
)

from intelligence_layer.prompt_template import (
    PromptTemplate,
)
from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from typing import List 
from pydantic import BaseModel


class MultipleChunkQaInput(BaseModel):
    chunks: List[str]
    question: str


class Source(BaseModel):
    text: str
    highlights: list[str]

class MultipleChunkQaOutput(BaseModel):
    answer: Optional[str]
    sources: Sequence[Source]

class MultipleChunkQa(Task[MultipleChunkQaInput, MultipleChunkQaOutput]):
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
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.completion = Completion(client)
        self.single_chunk_qa = SingleChunkQa(client, model)
        self.model = model

    def _format_prompt(self, question: str, answers: Sequence[str]) -> Prompt:
        template = PromptTemplate(self.PROMPT_TEMPLATE)
        return template.to_prompt(question=question, answers=answers)

    def _complete(self, prompt: Prompt, logger: DebugLogger) -> CompletionOutput:
        request = CompletionRequest(prompt)
        return self.completion.run(
            CompletionInput(request=request, model=self.model), logger
        )
    
    def run(
        self, input: MultipleChunkQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        qa_outputs: Sequence[SingleChunkQaOutput] = [
            self.single_chunk_qa.run(
                SingleChunkQaInput(question=input.question, chunk=chunk),
                logger.child_logger("Single Chunk QA"),
            )
            for chunk in input.chunks
        ]
        sources = [Source(text=chunk, highlights=qa_output.highlights) for qa_output, chunk in zip(qa_outputs, input.chunks)]

        logger.log("Intermediate Answers", [output.answer for output in qa_outputs])

        answers: List[str] = [
            output.answer for output in qa_outputs if output.answer is not None
        ]

        if len(answers) == 0:
            return MultipleChunkQaOutput(
                answer=None,
                sources=[],
            )

        prompt_text = self._format_prompt(input.question, answers)
        output = self._complete(prompt_text, logger.child_logger("Merge Answers"))

        return MultipleChunkQaOutput(
            answer=output.completion().strip(),
            sources=sources
        )
