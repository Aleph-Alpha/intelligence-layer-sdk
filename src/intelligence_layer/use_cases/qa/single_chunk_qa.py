from typing import Mapping, Optional, Sequence

from liquid import Template
from pydantic import BaseModel

from intelligence_layer.core import (
    CompleteInput,
    CompleteOutput,
    ControlModel,
    Language,
    LuminousControlModel,
    RichPrompt,
    Task,
    TaskSpan,
    TextChunk,
    TextHighlight,
    TextHighlightInput,
    TextHighlightOutput,
)
from intelligence_layer.core.prompt_template import TextCursor
from intelligence_layer.core.text_highlight import ScoredTextHighlight


class QaSetup(BaseModel):
    unformatted_instruction: str
    no_answer_str: str
    no_answer_logit_bias: Optional[float] = None


QA_INSTRUCTIONS = {
    Language("en"): QaSetup(
        unformatted_instruction='Question: {{question}}\nAnswer the question on the basis of the text. If there is no answer within the text, respond "{{no_answer_text}}".',
        no_answer_str="no answer in text",
        no_answer_logit_bias=1.0,
    ),
    Language("de"): QaSetup(
        unformatted_instruction='Beantworte die Frage anhand des Textes. Wenn sich die Frage nicht mit dem Text beantworten lässt, antworte "{{no_answer_text}}".\nFrage: {{question}}',
        no_answer_str="Unbeantwortbar",
        no_answer_logit_bias=0.5,
    ),
    Language("fr"): QaSetup(
        unformatted_instruction="{{question}}\nS'il n'y a pas de réponse, dites \"{{no_answer_text}}\". Ne répondez à la question qu'en vous basant sur le texte.",
        no_answer_str="pas de réponse dans le texte",
    ),
    Language("es"): QaSetup(
        unformatted_instruction='{{question}}\nSi no hay respuesta, di "{{no_answer_text}}". Responde sólo a la pregunta basándote en el texto.',
        no_answer_str="no hay respuesta en el texto",
    ),
    Language("it"): QaSetup(
        unformatted_instruction='{{question}}\nSe non c\'è risposta, dire "{{no_answer_text}}". Rispondere alla domanda solo in base al testo.',
        no_answer_str="nessuna risposta nel testo",
    ),
}


class SingleChunkQaInput(BaseModel):
    """The input for a `SingleChunkQa` task.

    Attributes:
        chunk: The (short) text to be asked about. Usually measures one or a few paragraph(s).
            Can't be longer than the context length of the model used minus the size of the system prompt.
        question: The question to be asked by about the chunk.
        language: The desired language of the answer. ISO 619 str with language e.g. en, fr, etc.
    """

    chunk: TextChunk
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
    highlights: Sequence[ScoredTextHighlight]


class SingleChunkQa(Task[SingleChunkQaInput, SingleChunkQaOutput]):
    """Answer a question on the basis of one chunk.

    Uses Aleph Alpha models to generate a natural language answer for a text chunk given a question.
    Will answer `None` if the language model determines that the question cannot be answered on the
    basis of the text.

    Args:
        model: The model used throughout the task for model related API calls.
        text_highlight: The task that is used for highlighting that parts of the input that are
            relevant for the answer. Defaults to :class:`TextHighlight` .
        instruction_config: defines instructions for different langaugaes.
        maximum_token: the maximal number of tokens to be generated for an answer.
    Attributes:
        NO_ANSWER_STR: The string to be generated by the model in case no answer can be found.


    Example:
        >>> import os
        >>> from intelligence_layer.core import Language, InMemoryTracer
        >>> from intelligence_layer.core import TextChunk
        >>> from intelligence_layer.use_cases import SingleChunkQa, SingleChunkQaInput
        >>>
        >>> task = SingleChunkQa()
        >>> input = SingleChunkQaInput(
        ...     chunk=TextChunk("Tina does not like pizza. However, Mike does."),
        ...     question="Who likes pizza?",
        ...     language=Language("en"),
        ... )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    def __init__(
        self,
        model: ControlModel | None = None,
        text_highlight: Task[TextHighlightInput, TextHighlightOutput] | None = None,
        instruction_config: Mapping[Language, QaSetup] = QA_INSTRUCTIONS,
        maximum_tokens: int = 64,
    ):
        super().__init__()
        self._model = model or LuminousControlModel("luminous-supreme-control-20240215")
        self._text_highlight = text_highlight or TextHighlight(
            LuminousControlModel("luminous-base-control-20240215")
        )
        self._instruction_config = instruction_config
        self._maximum_tokens = maximum_tokens

    def do_run(
        self, input: SingleChunkQaInput, task_span: TaskSpan
    ) -> SingleChunkQaOutput:
        qa_setup = input.language.language_config(self._instruction_config)

        instruction = Template(qa_setup.unformatted_instruction).render(
            question=input.question, no_answer_text=qa_setup.no_answer_str
        )

        no_answer_logit_bias = (
            self._get_no_answer_logit_bias(
                qa_setup.no_answer_str, qa_setup.no_answer_logit_bias
            )
            if qa_setup.no_answer_logit_bias
            else None
        )
        output, prompt = self._generate_answer(
            instruction,
            input.chunk,
            no_answer_logit_bias,
            task_span,
        )

        answer = self._no_answer_to_none(
            output.completion.strip(), qa_setup.no_answer_str
        )

        raw_highlights = (
            self._get_highlights(
                prompt,
                output.completion,
                task_span,
            )
            if answer
            else []
        )
        highlights = self._shift_highlight_ranges_to_input(prompt, raw_highlights)

        return SingleChunkQaOutput(
            answer=answer,
            highlights=highlights,
        )

    def _shift_highlight_ranges_to_input(
        self, prompt: RichPrompt, raw_highlights: Sequence[ScoredTextHighlight]
    ) -> Sequence[ScoredTextHighlight]:
        # This only works with models that have an 'input' range, e.g. control models.
        input_cursor = prompt.ranges["input"][0].start
        assert isinstance(input_cursor, TextCursor)
        input_offset = input_cursor.position
        return [
            ScoredTextHighlight(
                start=raw.start - input_offset,
                end=raw.end - input_offset,
                score=raw.score,
            )
            for raw in raw_highlights
        ]

    def _get_no_answer_logit_bias(
        self, no_answer_str: str, no_answer_logit_bias: float
    ) -> dict[int, float]:
        return {self._model.tokenize(no_answer_str).ids[0]: no_answer_logit_bias}

    def _generate_answer(
        self,
        instruction: str,
        input: str,
        no_answer_logit_bias: Optional[dict[int, float]],
        task_span: TaskSpan,
    ) -> tuple[CompleteOutput, RichPrompt]:
        prompt = self._model.to_instruct_prompt(instruction, input)

        return (
            self._model.complete(
                CompleteInput(
                    prompt=prompt,
                    maximum_tokens=self._maximum_tokens,
                    logit_bias=no_answer_logit_bias,
                ),
                task_span,
            ),
            prompt,
        )

    def _get_highlights(
        self,
        rich_prompt: RichPrompt,
        completion: str,
        task_span: TaskSpan,
    ) -> Sequence[ScoredTextHighlight]:
        highlight_input = TextHighlightInput(
            rich_prompt=rich_prompt,
            target=completion,
            focus_ranges=frozenset({"input"}),
        )
        highlight_output = self._text_highlight.run(highlight_input, task_span)
        return [h for h in highlight_output.highlights if h.score > 0]

    def _no_answer_to_none(self, completion: str, no_answer_str: str) -> Optional[str]:
        return completion if no_answer_str not in completion else None
