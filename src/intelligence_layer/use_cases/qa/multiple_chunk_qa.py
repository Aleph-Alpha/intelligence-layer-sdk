from typing import Iterable, Mapping, Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.complete import Instruct, InstructInput, PromptOutput
from intelligence_layer.core.detect_language import Language, language_config
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan
from intelligence_layer.use_cases.qa.single_chunk_qa import (
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


class MergeAnswersInstructConfig(BaseModel):
    instruction: str
    question_label: str
    answers_label: str
    final_answer_label: str


MERGE_ANSWERS_INSTRUCT_CONFIGS = {
    Language("en"): MergeAnswersInstructConfig(
        instruction="You will be given a number of Answers to a Question. Based on them, generate a single final answer. "
        "Condense multiple answers into a single answer. Rely only on the provided answers. Don't use the world's knowledge. "
        "The answer should combine the individual answers. If the answers contradict each other, e.g., one saying that the colour is green "
        "and the other saying that the colour is black, say that there are contradicting answers saying the colour is green or the colour is black.",
        question_label="Question",
        answers_label="Answers",
        final_answer_label="Final answer:",
    ),
    Language("it"): MergeAnswersInstructConfig(
        instruction="Vi verranno fornite diverse risposte a una domanda. Sulla base di queste, generate una singola risposta finale. "
        "Riunire più risposte in un'unica risposta. Basatevi solo sulle risposte fornite. Non utilizzate le conoscenze del mondo. "
        "La risposta deve combinare le singole risposte. Se le risposte si contraddicono, ad esempio una dice che il colore è verde e "
        "l'altra dice che il colore è nero, dire che ci sono risposte contraddittorie che dicono che il colore è verde o che il colore è nero.",
        question_label="Domanda",
        answers_label="Risposte",
        final_answer_label="Risposta finale:",
    ),
    Language("fr"): MergeAnswersInstructConfig(
        instruction="Vous recevrez un certain nombre de réponses à une question. Sur la base de ces réponses, générez une seule réponse finale. "
        "Condenser plusieurs réponses en une seule. Ne vous fiez qu'aux réponses fournies. N'utilisez pas les connaissances du monde entier. "
        "La réponse doit combiner les différentes réponses. Si les réponses se contredisent, par exemple si l'une dit que la couleur est verte et "
        "l'autre que la couleur est noire, dites qu'il y a des réponses contradictoires disant que la couleur est verte ou que la couleur est noire.",
        question_label="Question",
        answers_label="Réponses",
        final_answer_label="Réponse finale:",
    ),
    Language("de"): MergeAnswersInstructConfig(
        instruction="Sie erhalten eine Reihe von Antworten auf eine Frage. Erstellen Sie auf dieser Grundlage eine einzige endgültige Antwort. "
        "Fassen Sie mehrere Antworten zu einer einzigen Antwort zusammen. Verlassen Sie sich nur auf die vorgegebenen Antworten. "
        "Verwenden Sie nicht das Wissen der Welt. Die Antwort sollte die einzelnen Antworten kombinieren. Wenn sich die Antworten widersprechen, "
        "z. B. wenn eine Antwort besagt, dass die Farbe grün ist, und die andere, dass die Farbe schwarz ist, sagen Sie, "
        "dass es widersprüchliche Antworten gibt, die besagen, dass die Farbe grün oder die Farbe schwarz ist.",
        question_label="Frage",
        answers_label="Antworten",
        final_answer_label="Endgültige Antwort:",
    ),
    Language("es"): MergeAnswersInstructConfig(
        instruction="Se le darán varias respuestas a una pregunta. A partir de ellas, genere una única respuesta final. "
        "Condensar varias respuestas en una sola. Apóyate únicamente en las respuestas proporcionadas. No utilice el conocimiento del mundo. "
        "La respuesta debe combinar las respuestas individuales. Si las respuestas se contradicen, por ejemplo, una dice que el color es verde "
        "y la otra que el color es negro, di que hay respuestas contradictorias que dicen que el color es verde o que el color es negro.",
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
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        MERGE_ANSWERS_INSTRUCTION: The instruction template used for combining multiple answers into one.

    Example:
        >>> import os
        >>> from intelligence_layer.connectors import (
        ...     LimitedConcurrencyClient,
        ... )
        >>> from intelligence_layer.core import Language
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.core.chunk import Chunk
        >>> from intelligence_layer.use_cases import (
        ...     MultipleChunkQa,
        ...     MultipleChunkQaInput,
        ... )


        >>> client = LimitedConcurrencyClient.from_token(os.getenv("AA_TOKEN"))
        >>> task = MultipleChunkQa(client)
        >>> input = MultipleChunkQaInput(
        ...     chunks=[Chunk("Tina does not like pizza."), Chunk("Mike is a big fan of pizza.")],
        ...     question="Who likes pizza?",
        ...     language=Language("en"),
        ... )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
        >>> print(output.answer)
        Mike likes pizza.
    """

    MERGE_ANSWERS_INSTRUCTION = """You will be given a number of Answers to a Question. Based on them, generate a single final answer.
Condense multiple answers into a single answer. Rely only on the provided answers. Don't use the world's knowledge. The answer should combine the individual answers. If the answers contradict each other, e.g., one saying that the colour is green and the other saying that the colour is black, say that there are contradicting answers saying the colour is green or the colour is black."""

    def __init__(
        self,
        client: AlephAlphaClientProtocol,
        model: str = "luminous-supreme-control",
        merge_answers_instruct_configs: Optional[
            Mapping[Language, MergeAnswersInstructConfig]
        ] = None,
    ):
        super().__init__()
        self._client = client
        self._instruction = Instruct(client, model)
        self._single_chunk_qa = SingleChunkQa(client, model)
        self._model = model
        self._merge_answers_instruct_configs = (
            merge_answers_instruct_configs or MERGE_ANSWERS_INSTRUCT_CONFIGS
        )

    def do_run(
        self, input: MultipleChunkQaInput, task_span: TaskSpan
    ) -> MultipleChunkQaOutput:
        instruct_config = language_config(
            input.language, self._merge_answers_instruct_configs
        )

        qa_outputs = self._single_chunk_qa.run_concurrently(
            (
                SingleChunkQaInput(
                    question=input.question, chunk=chunk, language=input.language
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
                for qa_output, chunk in zip(qa_outputs, input.chunks)
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
    ) -> PromptOutput:
        return self._instruction.run(
            InstructInput(
                instruction=instruction_config.instruction,
                input=input,
                response_prefix=f"\n{instruction_config.final_answer_label}",
            ),
            task_span,
        )
