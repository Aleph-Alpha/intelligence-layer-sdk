from typing import Iterable, Optional, Sequence

from aleph_alpha_client import Client
from pydantic import BaseModel

from intelligence_layer.core.complete import (
    Instruct,
    InstructInput,
    PromptOutput,
)
from intelligence_layer.core.detect_language import Language
from intelligence_layer.use_cases.qa.single_chunk_qa import (
    SingleChunkQaInput,
    SingleChunkQaOutput,
    SingleChunkQa,
)
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.task import Task
from intelligence_layer.core.logger import DebugLogger


class MultipleChunkQaInput(BaseModel):
    """The input for a `MultipleChunkQa` task.

    Attributes:
        chunks: The list of chunks that will be used to answer the question.
            Can be arbitrarily long list of chunks.
        question: The question that will be answered based on the chunks.
        language: The desired language of the answer. ISO 619 str with language e.g. en, fr, etc.
    """

    chunks: Sequence[Chunk]
    question: str
    language: Language = Language("en")


class Subanswer(BaseModel):
    """Individual answer based on just one of the multiple chunks.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        chunk: Piece of the original text that answer is based on.
        highlights: The specific sentences that explain the answer the most.
            These are generated by the `TextHighlight` Task.
    """

    answer: str
    chunk: Chunk
    highlights: Sequence[str]


class MultipleChunkQaOutput(BaseModel):
    """The output of a `MultipleChunkQa` task.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        subanswers: All the subanswers used to generate the answer.
    """

    answer: Optional[str]
    subanswers: Sequence[Subanswer]


class MultipleChunkQa(Task[MultipleChunkQaInput, MultipleChunkQaOutput]):
    """Answer a question on the basis of a list of text chunks.

    Uses Aleph Alpha models to generate a natural language answer based on multiple text chunks.
    Best for longer texts that are already split into smaller units (chunks).
    Relies on SingleChunkQa to generate answers for each chunk and then merges the answers into a single final answer.
    Includes logic to return 'answer = None' if the language model determines that the question cannot be
    reliably answered on the basis of the chunks.

    Note:
        `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        MERGE_ANSWERS_INSTRUCTION: The instruction template used for combining multiple answers into one.

    Example:
        >>> client = Client(token="AA_TOKEN")
        >>> task = MultipleChunkQa(client)
        >>> input = MultipleChunkQaInput(
                chunks=["Tina does not like pizza.", "Mike is a big fan of pizza."],
                question="Who likes pizza?",
                language=Language("en")
            )
        >>> logger = InMemoryLogger(name="Multiple Chunk QA")
        >>> output = task.run(input, logger)
        >>> print(output.answer)
        Mike likes pizza.
    """

    MERGE_ANSWERS_INSTRUCTION = """You will be given a number of Answers to a Question. Based on them, generate a single final answer.
Condense multiple answers into a single answer. Rely only on the provided answers. Don't use the world's knowledge. The answer should combine the individual answers. If the answers contradict each other, e.g., one saying that the colour is green and the other saying that the colour is black, say that there are contradicting answers saying the colour is green or the colour is black."""

    def __init__(
        self,
        client: Client,
        model: str = "luminous-supreme-control",
    ):
        super().__init__()
        self._client = client
        self._instruction = Instruct(client)
        self._single_chunk_qa = SingleChunkQa(client, model)
        self._model = model

    def run(
        self, input: MultipleChunkQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        qa_outputs = self._single_chunk_qa.run_concurrently(
            (
                SingleChunkQaInput(
                    question=input.question, chunk=chunk, language=input.language
                )
                for chunk in input.chunks
            ),
            logger,
        )
        final_answer = self._merge_answers(input.question, qa_outputs, logger)

        return MultipleChunkQaOutput(
            answer=final_answer,
            subanswers=[
                Subanswer(
                    answer=qa_output.answer,
                    chunk=chunk,
                    highlights=qa_output.highlights,
                )
                for qa_output, chunk in zip(qa_outputs, input.chunks)
                if qa_output.answer
            ],
        )

    def _merge_answers(
        self,
        question: str,
        qa_outputs: Iterable[SingleChunkQaOutput],
        logger: DebugLogger,
    ) -> Optional[str]:
        answers = [output.answer for output in qa_outputs if output.answer]
        if len(answers) == 0:
            return None
        elif len(answers) == 1:
            return answers[0]

        joined_answers = "\n".join(answers)
        return self._instruct(
            f"""Question: {question}

Answers:
{joined_answers}""",
            logger,
        ).response

    def _instruct(self, input: str, logger: DebugLogger) -> PromptOutput:
        return self._instruction.run(
            InstructInput(
                instruction=self.MERGE_ANSWERS_INSTRUCTION,
                input=input,
                model=self._model,
                response_prefix="\nFinal answer:",
            ),
            logger,
        )
