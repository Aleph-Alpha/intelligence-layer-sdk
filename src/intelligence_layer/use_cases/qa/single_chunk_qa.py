from typing import Mapping, Optional, Sequence

from aleph_alpha_client import Client
from liquid import Template
from pydantic import BaseModel

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.complete import Instruct, InstructInput, PromptOutput
from intelligence_layer.core.detect_language import Language, language_config
from intelligence_layer.core.prompt_template import PromptWithMetadata
from intelligence_layer.core.task import Task
from intelligence_layer.core.text_highlight import TextHighlight, TextHighlightInput
from intelligence_layer.core.tracer import TaskSpan

QA_INSTRUCTIONS = {
    Language(
        "en"
    ): """{{question}} If there's no answer, say {{no_answer_text}}. Only answer the question based on the text.""",
    Language(
        "de"
    ): """{{question}} Wenn es keine Antwort gibt, gib {{no_answer_text}} aus. Beantworte die Frage nur anhand des Textes.""",
    Language(
        "fr"
    ): """{{question}} S'il n'y a pas de réponse, dites {{no_answer_text}}. Ne répondez à la question qu'en vous basant sur le texte. """,
    Language(
        "es"
    ): """{{question}}Si no hay respuesta, di {{no_answer_text}}. Responde sólo a la pregunta basándote en el texto.""",
    Language(
        "it"
    ): """{{question}}Se non c'è risposta, dire {{no_answer_text}}. Rispondere alla domanda solo in base al testo.""",
}


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
        instruction_config: Mapping[Language, str] = QA_INSTRUCTIONS,
    ):
        super().__init__()
        self._client = client
        self._model = model
        self._instruction = Instruct(client)
        self._text_highlight = TextHighlight(client)
        self._instruction_config = instruction_config

    def do_run(
        self, input: SingleChunkQaInput, task_span: TaskSpan
    ) -> SingleChunkQaOutput:
        instruction_text = language_config(input.language, self._instruction_config)

        output = self._generate_answer(
            Template(instruction_text).render(
                question=input.question, no_answer_text=self.NO_ANSWER_STR
            ),
            input.chunk,
            task_span,
        )
        answer = self._no_answer_to_none(output.response.strip())
        highlights = (
            self._get_highlights(
                output.prompt_with_metadata,
                output.response,
                task_span,
            )
            if answer
            else []
        )
        return SingleChunkQaOutput(
            answer=answer,
            highlights=highlights,
        )

    def _generate_answer(
        self, instruction: str, input: str, task_span: TaskSpan
    ) -> PromptOutput:
        return self._instruction.run(
            InstructInput(instruction=instruction, input=input, model=self._model),
            task_span,
        )

    def _get_highlights(
        self,
        prompt_with_metadata: PromptWithMetadata,
        completion: str,
        task_span: TaskSpan,
    ) -> Sequence[str]:
        highlight_input = TextHighlightInput(
            prompt_with_metadata=prompt_with_metadata,
            target=completion,
            model=self._model,
            focus_ranges=frozenset({"input"}),
        )
        highlight_output = self._text_highlight.run(highlight_input, task_span)
        return [h.text for h in highlight_output.highlights if h.score > 0]

    def _no_answer_to_none(self, completion: str) -> Optional[str]:
        return completion if completion != self.NO_ANSWER_STR else None
