from typing import Generic, Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import (
    ID,
    BaseRetriever,
    SearchResult,
)
from intelligence_layer.core.chunk import TextChunk
from intelligence_layer.core.task import Task
from intelligence_layer.core.text_highlight import ScoredTextHighlight
from intelligence_layer.core.tracer.tracer import TaskSpan
from intelligence_layer.use_cases.search.search import Search, SearchInput

from .retriever_based_qa import RetrieverBasedQaInput
from .single_chunk_qa import SingleChunkQa, SingleChunkQaInput, SingleChunkQaOutput


class AnswerSource(BaseModel, Generic[ID]):
    chunk: SearchResult[ID]
    highlights: Sequence[ScoredTextHighlight]


class MulMultipleChunkRetrieverQaOutput(BaseModel, Generic[ID]):
    answer: Optional[str]
    sources: Sequence[AnswerSource[ID]]


class MultipleChunkRetrieverQa(
    Task[RetrieverBasedQaInput, MulMultipleChunkRetrieverQaOutput[ID]], Generic[ID]
):
    """Answer a question based on documents found by a retriever.

    `MultipleChunkRetrieverBasedQa` is a task that answers a question based on a set of documents.
    It relies on some retriever of type `BaseRetriever` that has the ability to access texts.
    In contrast to the regular `RetrieverBasedQa`, this tasks injects multiple chunks into one
    `SingleChunkQa` task run.

    Note:
        `model` provided should be a control-type model.

    Args:
        retriever: Used to access and return a set of texts.
        k: number of top chunk search results to inject into :class:`SingleChunkQa`-task.
        qa_task: The task that is used to generate an answer for a single chunk (retrieved through
            the retriever). Defaults to :class:`SingleChunkQa`.

    Example:
        >>> import os
        >>> from intelligence_layer.connectors import DocumentIndexClient
        >>> from intelligence_layer.connectors import DocumentIndexRetriever
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.use_cases import MultipleChunkRetrieverQa, RetrieverBasedQaInput


        >>> token = os.getenv("AA_TOKEN")
        >>> document_index = DocumentIndexClient(token)
        >>> retriever = DocumentIndexRetriever(document_index, "aleph-alpha", "wikipedia-de", 3)
        >>> task = MultipleChunkRetrieverQa(retriever, k=2)
        >>> input_data = RetrieverBasedQaInput(question="When was Rome founded?")
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input_data, tracer)
    """

    def __init__(
        self,
        retriever: BaseRetriever[ID],
        k: int = 5,
        single_chunk_qa: Task[SingleChunkQaInput, SingleChunkQaOutput] | None = None,
    ):
        super().__init__()
        self._search = Search(retriever)
        self._k = k
        self._single_chunk_qa = single_chunk_qa or SingleChunkQa()

    @staticmethod
    def _combine_input_texts(chunks: Sequence[str]) -> tuple[TextChunk, Sequence[int]]:
        start_indices: list[int] = []
        combined_text = ""
        for chunk in chunks:
            start_indices.append(len(combined_text))
            combined_text += chunk + "\n\n"
        return (TextChunk(combined_text.strip()), start_indices)

    @staticmethod
    def _get_highlights_per_chunk(
        chunk_start_indices: Sequence[int], highlights: Sequence[ScoredTextHighlight]
    ) -> Sequence[Sequence[ScoredTextHighlight]]:
        overlapping_ranges = []
        for i in range(len(chunk_start_indices)):
            current_start = chunk_start_indices[i]
            next_start = (
                chunk_start_indices[i + 1]
                if i + 1 < len(chunk_start_indices)
                else float("inf")
            )

            current_overlaps = []
            for highlight in highlights:
                if highlight.start < next_start and highlight.end > current_start:
                    highlights_with_indices_fixed = ScoredTextHighlight(
                        start=max(0, highlight.start - current_start),
                        end=min(highlight.end - current_start, next_start),
                        score=highlight.score,
                    )
                    current_overlaps.append(highlights_with_indices_fixed)

            overlapping_ranges.append(current_overlaps)
        return overlapping_ranges

    def do_run(
        self, input: RetrieverBasedQaInput, task_span: TaskSpan
    ) -> MulMultipleChunkRetrieverQaOutput[ID]:
        search_output = self._search.run(
            SearchInput(query=input.question), task_span
        ).results
        sorted_search_output = sorted(
            search_output,
            key=lambda output: output.score,  # not reversing on purpose because model performs better if relevant info is at the end
        )[-self._k :]

        chunk, chunk_start_indices = self._combine_input_texts(
            [output.document_chunk.text for output in sorted_search_output]
        )

        single_chunk_qa_input = SingleChunkQaInput(
            chunk=chunk,
            question=input.question,
            language=input.language,
        )

        single_chunk_qa_output = self._single_chunk_qa.run(
            single_chunk_qa_input, task_span
        )

        highlights_per_chunk = self._get_highlights_per_chunk(
            chunk_start_indices, single_chunk_qa_output.highlights
        )

        return MulMultipleChunkRetrieverQaOutput(
            answer=single_chunk_qa_output.answer,
            sources=[
                AnswerSource(
                    chunk=chunk,
                    highlights=highlights,
                )
                for chunk, highlights in zip(sorted_search_output, highlights_per_chunk)
            ],
        )
