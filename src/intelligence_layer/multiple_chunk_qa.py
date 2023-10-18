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
from pydantic import BaseModel


class MultipleChunkQaInput(BaseModel):
    """Input for a multiple chunk QA task.
    
    Attributes:
        chunks: list of chunks that will be used to answer the question.
            This can be an arbitrarily long list of chunks.
        question: The question that will be answered based on the chunks.
    """
    chunks: list[str]
    question: str


class Source(BaseModel):
    """Source for the multiple chunk QA output.
    
    Attributes:
        text: Piece of the original text that the qa output answer is based on.
        highlights: The specific sentences that explain the answer the most. 
            These are generated by the TextHighlight Task. 
    """
    text: str
    highlights: Sequence[str]


class MultipleChunkQaOutput(BaseModel):
    """Multiple chunk qa output.
    
    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        sources: All the sources used to generate the answer. 
    """
    answer: Optional[str]
    sources: Sequence[Source]


class MultipleChunkQa(Task[MultipleChunkQaInput, MultipleChunkQaOutput]):
    """Task implementation for answering a question based on multiple chunks.

    Uses Aleph Alpha models to generate a natural language answer based on multiple text chunks.
    Use this instead of SingleChunkQa if the texts you would like to ask about are larger than the models context size.
    This task relies on SingleChunkQa to generate answers based on chunks and then merges the answers into a single final answer.

    Includes logic to return 'answer = None' if the language model determines that the question cannot be answered on the basis of the chunks.

    Note:
        'model' provided must be a control-type model for the prompt to function as expected.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        PROMPT_TEMPLATE_STR: The prompt template used for answering the question. 
            'chunk' and 'question' will be inserted here. 
        NO_ANSWER_STR: The string to be generated by the model in case no answer can be found.

    Example:
        >>> client = Client(token="AA_TOKEN")
        >>> task = MultipleChunkQa(client)
        >>> input = MultipleChunkQaInput(
        >>>     chunks=["Tina does not like pizza.", "However, Mike does."],
        >>>     question="Who likes pizza?"
        >>> )
        >>> logger = InMemoryLogger(name="QA")
        >>> output = task.run(input, logger)
        >>> print(output.answer)
        Mike likes pizza.
    """

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
                logger.child_logger(f"Single Chunk QA for Chunk {chunk_number}"),
            )
            for chunk_number, chunk in enumerate(input.chunks)
        ]

        answers: list[str] = [
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
            sources=[
                Source(text=chunk, highlights=qa_output.highlights)
                for qa_output, chunk in zip(qa_outputs, input.chunks)
            ],
        )
