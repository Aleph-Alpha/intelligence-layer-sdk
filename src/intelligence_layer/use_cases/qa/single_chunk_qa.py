from typing import Optional, Sequence

from aleph_alpha_client import Client
from liquid import Template
from pydantic import BaseModel

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.complete import Instruct, InstructInput, PromptOutput
from intelligence_layer.core.detect_language import Language, LanguageNotSupportedError
from intelligence_layer.core.prompt_template import PromptWithMetadata
from intelligence_layer.core.task import Task
from intelligence_layer.core.text_highlight import TextHighlight, TextHighlightInput
from intelligence_layer.core.tracer import Tracer
from intelligence_layer.use_cases.qa.luminous_prompts import (
    LANGUAGES_QA_INSTRUCTIONS as LUMINOUS_LANGUAGES_QA_INSTRUCTIONS,
)


class SingleChunkQaInput(BaseModel):
    """The input for a `SingleChunkQa` task.

    Attributes:
        chunk: The (short) text to be asked about. Usually measures one or a few paragraph(s).
            Can't be longer than the context length of the model used minus the size of the system prompt.
        question: The question to be asked by about the chunk.
        language: The desired language of the answer. ISO 619 str with language e.g. en, fr, etc.
    """

    chunk: Chunk
    question: str
    language: Language = Language("en")


class SingleChunkQaOutput(BaseModel):
    """The output of a `SingleChunkQa` task.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        highlights: Highlights indicating which parts of the chunk contributed to the answer.
            Each highlight is a quote from the text.
    """

    answer: Optional[str]
    highlights: Sequence[str]


class SingleChunkQa(Task[SingleChunkQaInput, SingleChunkQaOutput]):
    """Answer a question on the basis of one chunk.

    Uses Aleph Alpha models to generate a natural language answer for a text chunk given a question.
    Will answer `None` if the language model determines that the question cannot be answered on the
    basis of the text.

    Note:
        `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        PROMPT_TEMPLATE_STR: The prompt template used for answering the question.
            Includes liquid logic interpreted by 'PromptTemplate' specifically for generating
            explainability-based highlights using `TextHighlight`.
        NO_ANSWER_STR: The string to be generated by the model in case no answer can be found.


    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = SingleChunkQa(client)
        >>> input = SingleChunkQaInput(
                chunk="Tina does not like pizza. However, Mike does.",
                question="Who likes pizza?",
                language=Language("en")
            )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
        >>> print(output.answer)
        Mike likes pizza.
    """

    NO_ANSWER_STR = "NO_ANSWER_IN_TEXT"

    def __init__(
        self,
        client: Client,
        model: str = "luminous-supreme-control",
    ):
        super().__init__()
        self._client = client
        self._model = model
        self._instruction = Instruct(client)
        self._text_highlight = TextHighlight(client)

    def run(self, input: SingleChunkQaInput, tracer: Tracer) -> SingleChunkQaOutput:
        try:
            prompt = LUMINOUS_LANGUAGES_QA_INSTRUCTIONS[input.language]
        except KeyError:
            allowed_languages = list(LUMINOUS_LANGUAGES_QA_INSTRUCTIONS.keys())
            raise LanguageNotSupportedError(
                f"{input.language} not in allowed languages ({allowed_languages})"
            )

        output = self._instruct(
            Template(prompt).render(
                question=input.question, no_answer_text=self.NO_ANSWER_STR
            ),
            input.chunk,
            tracer,
        )
        answer = self._no_answer_to_none(output.response.strip())
        highlights = (
            self._get_highlights(
                output.prompt_with_metadata,
                output.response,
                tracer,
            )
            if answer
            else []
        )
        return SingleChunkQaOutput(
            answer=answer,
            highlights=highlights,
        )

    def _instruct(self, instruction: str, input: str, tracer: Tracer) -> PromptOutput:
        return self._instruction.run(
            InstructInput(instruction=instruction, input=input, model=self._model),
            tracer,
        )

    def _get_highlights(
        self,
        prompt_with_metadata: PromptWithMetadata,
        completion: str,
        tracer: Tracer,
    ) -> Sequence[str]:
        highlight_input = TextHighlightInput(
            prompt_with_metadata=prompt_with_metadata,
            target=completion,
            model=self._model,
            focus_ranges=frozenset({"input"}),
        )
        highlight_output = self._text_highlight.run(highlight_input, tracer)
        return [h.text for h in highlight_output.highlights if h.score > 0]

    def _no_answer_to_none(self, completion: str) -> Optional[str]:
        return completion if completion != self.NO_ANSWER_STR else None
