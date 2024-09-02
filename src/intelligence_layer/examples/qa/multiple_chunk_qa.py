from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

from pydantic import BaseModel

from intelligence_layer.core import (
    CompleteInput,
    CompleteOutput,
    ControlModel,
    Language,
    LuminousControlModel,
    Task,
    TaskSpan,
    TextChunk,
)
from intelligence_layer.core.text_highlight import ScoredTextHighlight
from intelligence_layer.examples.qa.single_chunk_qa import (
    SingleChunkQa,
    SingleChunkQaInput,
    SingleChunkQaOutput,
)


class MultipleChunkQaInput(BaseModel):
    """The input for a `MultipleChunkQa` task.

    Attributes:
        chunks: The list of chunks that will be used to answer the question.
            Can be arbitrarily long list of chunks.
        question: The question that will be answered based on the chunks.
        language: The desired language of the answer. ISO 619 str with language e.g. en, fr, etc.
        explainability_enabled: Whether to generate highlights (using the explainability feature) for the answer;  default False for performance reasons
    """

    chunks: Sequence[TextChunk]
    question: str
    language: Language = Language("en")
    explainability_enabled: bool = False


class Subanswer(BaseModel):
    """Individual answer based on just one of the multiple chunks.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        chunk: Piece of the original text that answer is based on.
        highlights: The specific sentences that explain the answer the most.
            These are generated by the `TextHighlight` Task.
    """

    answer: Optional[str]
    chunk: TextChunk
    highlights: Sequence[ScoredTextHighlight]


class MultipleChunkQaOutput(BaseModel):
    """The output of a `MultipleChunkQa` task.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        subanswers: All the subanswers used to generate the answer.
    """

    answer: Optional[str]
    subanswers: Sequence[Subanswer]


class MergeAnswersInstructConfig(BaseModel):
    instruction: str
    question_label: str
    answers_label: str
    final_answer_label: str
    maximum_tokens: int = 128


MERGE_ANSWERS_INSTRUCT_CONFIGS = {
    Language("en"): MergeAnswersInstructConfig(
        instruction="You are tasked with combining multiple answers into a single answer. "
        "If conflicting answers arise, acknowledge the discrepancies by presenting them collectively. "
        "Your answer should not be lomnger than 5 sentences.",
        question_label="Question",
        answers_label="Answers",
        final_answer_label="Final answer:",
    ),
    Language("it"): MergeAnswersInstructConfig(
        instruction="Il compito è quello di combinare più risposte in un'unica risposta. "
        "Se emergono risposte contrastanti, riconoscete le discrepanze presentandole collettivamente. "
        "La risposta non deve essere più lunga di 5 frasi.",
        question_label="Domanda",
        answers_label="Risposte",
        final_answer_label="Risposta finale:",
    ),
    Language("fr"): MergeAnswersInstructConfig(
        instruction="Vous devez combiner plusieurs réponses en une seule. "
        "Si des réponses contradictoires apparaissent, reconnaissez les divergences en les présentant collectivement. "
        "Votre réponse ne doit pas dépasser 5 phrases.",
        question_label="Question",
        answers_label="Réponses",
        final_answer_label="Réponse finale:",
    ),
    Language("de"): MergeAnswersInstructConfig(
        instruction="Fasse alle Antworten zu einer einzigen Antwort zusammen. "
        "Falls es Widersprüche gibt, präsentiere diese. "
        "Deine Antwort sollte nicht länger als 5 Sätze sein.",
        question_label="Frage",
        answers_label="Antworten",
        final_answer_label="Endgültige Antwort:",
    ),
    Language("es"): MergeAnswersInstructConfig(
        instruction="Su tarea consiste en combinar varias respuestas en una sola. "
        "Si surgen respuestas contradictorias, reconozca las discrepancias presentándolas colectivamente. "
        "Su respuesta no debe superar las 5 frases.",
        question_label="Pregunta",
        answers_label="Respuestas",
        final_answer_label="Respuesta final:",
    ),
}


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
        single_chunk_qa: The task that is used to generate an answer based on a single chunk.
            Defaults to :class:`SingleChunkQa` .
        model: The model used throughout the task for model related API calls.
            Defaults to luminous-supreme-control.
        merge_answers_instruct_configs: Mapping language used to prompt parameters.

    Example:
        >>> import os
        >>> from intelligence_layer.connectors import (
        ...     LimitedConcurrencyClient,
        ... )
        >>> from intelligence_layer.core import Language, InMemoryTracer
        >>> from intelligence_layer.core.chunk import TextChunk
        >>> from intelligence_layer.examples import (
        ...     MultipleChunkQa,
        ...     MultipleChunkQaInput,
        ... )


        >>> task = MultipleChunkQa()
        >>> input = MultipleChunkQaInput(
        ...     chunks=[TextChunk("Tina does not like pizza."), TextChunk("Mike is a big fan of pizza.")],
        ...     question="Who likes pizza?",
        ...     language=Language("en"),
        ... )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
        >>> print(output.answer)
        Mike likes pizza.
    """

    def __init__(
        self,
        single_chunk_qa: Task[SingleChunkQaInput, SingleChunkQaOutput] | None = None,
        merge_answers_model: ControlModel | None = None,
        merge_answers_instruct_configs: Mapping[
            Language, MergeAnswersInstructConfig
        ] = MERGE_ANSWERS_INSTRUCT_CONFIGS,
    ):
        super().__init__()
        self._model = merge_answers_model or LuminousControlModel(
            "luminous-supreme-control"
        )
        self._single_chunk_qa = single_chunk_qa or SingleChunkQa(self._model)
        self._merge_answers_instruct_configs = merge_answers_instruct_configs

    def do_run(
        self, input: MultipleChunkQaInput, task_span: TaskSpan
    ) -> MultipleChunkQaOutput:
        instruct_config = input.language.language_config(
            self._merge_answers_instruct_configs
        )

        qa_outputs = self._single_chunk_qa.run_concurrently(
            (
                SingleChunkQaInput(
                    question=input.question, chunk=chunk, language=input.language, explainability_enabled=input.explainability_enabled
                )
                for chunk in input.chunks
            ),
            task_span,
        )
        final_answer = self._merge_answers(
            input.question, qa_outputs, instruct_config, task_span
        )

        return MultipleChunkQaOutput(
            answer=final_answer,
            subanswers=[
                Subanswer(
                    answer=qa_output.answer,
                    chunk=chunk,
                    highlights=qa_output.highlights,
                )
                for qa_output, chunk in zip(qa_outputs, input.chunks, strict=True)
                if qa_output.answer
            ],
        )

    def _merge_answers(
        self,
        question: str,
        qa_outputs: Iterable[SingleChunkQaOutput],
        instruction_config: MergeAnswersInstructConfig,
        task_span: TaskSpan,
    ) -> Optional[str]:
        answers = [output.answer for output in qa_outputs if output.answer]
        if len(answers) == 0:
            return None
        elif len(answers) == 1:
            return answers[0]

        joined_answers = "\n".join(answers)
        return self._instruct(
            f"""{instruction_config.question_label}: {question}

{instruction_config.answers_label}:
{joined_answers}""",
            instruction_config,
            task_span,
        ).completion

    def _instruct(
        self,
        input: str,
        instruction_config: MergeAnswersInstructConfig,
        task_span: TaskSpan,
    ) -> CompleteOutput:
        prompt = self._model.to_instruct_prompt(
            instruction_config.instruction,
            input=input,
            response_prefix=f" {instruction_config.final_answer_label}",
        )
        return self._model.complete(
            CompleteInput(
                prompt=prompt, maximum_tokens=instruction_config.maximum_tokens
            ),
            task_span,
        )
