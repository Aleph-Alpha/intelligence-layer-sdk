from typing import Generic, Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import ID, BaseRetriever
from intelligence_layer.core import Language, Task, TaskSpan, TextChunk
from intelligence_layer.use_cases.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
    Subanswer,
)
from intelligence_layer.use_cases.search.search import Search, SearchInput


class RetrieverBasedQaInput(BaseModel):
    """The input for a `RetrieverBasedQa` task.

    Attributes:
        question: The question to be answered based on the documents accessed
            by the retriever.
        language: The desired language of the answer. ISO 619 str with language e.g. en, fr, etc.
    """

    question: str
    language: Language = Language("en")


class EnrichedSubanswer(Subanswer, Generic[ID]):
    """Individual answer for a chunk that also contains the origin of the chunk.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        chunk: Piece of the original text that answer is based on.
        highlights: The specific sentences that explain the answer the most.
            These are generated by the `TextHighlight` Task.
        id: The id of the document where the chunk came from.
    """

    id: ID


class RetrieverBasedQaOutput(BaseModel, Generic[ID]):
    """The output of a `RetrieverBasedQa` task.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        subanswers: All the subanswers used to generate the answer.
    """

    answer: Optional[str]
    subanswers: Sequence[EnrichedSubanswer[ID]]


class RetrieverBasedQa(
    Task[RetrieverBasedQaInput, RetrieverBasedQaOutput[ID]], Generic[ID]
):
    """Answer a question based on documents found by a retriever.

    `RetrieverBasedQa` is a task that answers a question based on a set of documents.
    Relies on some retriever of type `BaseRetriever` that has the ability to access texts.

    Note:
        `model` provided should be a control-type model.

    Args:
        retriever: Used to access and return a set of texts.
        qa_task: The task that is used to generate an answer for a single chunk (retrieved through
            the retriever). Defaults to :class:`MultipleChunkQa` .

    Example:
        >>> import os
        >>> from intelligence_layer.connectors import DocumentIndexClient
        >>> from intelligence_layer.connectors import DocumentIndexRetriever
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.use_cases import RetrieverBasedQa, RetrieverBasedQaInput


        >>> token = os.getenv("AA_TOKEN")
        >>> document_index = DocumentIndexClient(token)
        >>> retriever = DocumentIndexRetriever(document_index, "asymmetric", "aleph-alpha", "wikipedia-de", 3)
        >>> task = RetrieverBasedQa(retriever)
        >>> input_data = RetrieverBasedQaInput(question="When was Rome founded?")
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input_data, tracer)
    """

    def __init__(
        self,
        retriever: BaseRetriever[ID],
        multi_chunk_qa: Task[MultipleChunkQaInput, MultipleChunkQaOutput] | None = None,
    ):
        super().__init__()
        self._search = Search(retriever)
        self._multi_chunk_qa = multi_chunk_qa or MultipleChunkQa()

    def do_run(
        self, input: RetrieverBasedQaInput, task_span: TaskSpan
    ) -> RetrieverBasedQaOutput[ID]:
        search_output = self._search.run(
            SearchInput(query=input.question), task_span
        ).results
        sorted_search_output = sorted(
            search_output, key=lambda output: output.score, reverse=True
        )

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[
                TextChunk(output.document_chunk.text) for output in sorted_search_output
            ],
            question=input.question,
            language=input.language,
        )

        multi_chunk_qa_output = self._multi_chunk_qa.run(
            multi_chunk_qa_input, task_span
        )

        enriched_answers = [
            EnrichedSubanswer(
                answer=answer.answer,
                chunk=TextChunk(input.document_chunk.text),
                highlights=answer.highlights,
                id=input.id,
            )
            for answer, input in zip(
                multi_chunk_qa_output.subanswers, sorted_search_output
            )
        ]
        correctly_formatted_output = RetrieverBasedQaOutput(
            answer=multi_chunk_qa_output.answer,
            subanswers=enriched_answers,
        )
        return correctly_formatted_output
