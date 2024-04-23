from collections import defaultdict
from copy import deepcopy
from typing import Generic, Optional, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import (
    ID,
    BaseRetriever,
    SearchResult,
)
from intelligence_layer.core.chunk import TextChunk
from intelligence_layer.core.model import ControlModel, LuminousControlModel
from intelligence_layer.core.task import Task
from intelligence_layer.core.text_highlight import ScoredTextHighlight
from intelligence_layer.core.tracer.tracer import TaskSpan
from intelligence_layer.use_cases.search.expand_chunks import (
    ExpandChunks,
    ExpandChunksInput,
)
from intelligence_layer.use_cases.search.search import Search, SearchInput

from .retriever_based_qa import RetrieverBasedQaInput
from .single_chunk_qa import SingleChunkQa, SingleChunkQaInput, SingleChunkQaOutput


class EnrichedChunk(BaseModel, Generic[ID]):
    document_id: ID
    chunk: TextChunk
    indices: tuple[int, int]


class AnswerSource(BaseModel, Generic[ID]):
    chunk: EnrichedChunk[ID]
    highlights: Sequence[ScoredTextHighlight]


class MultipleChunkRetrieverQaOutput(BaseModel, Generic[ID]):
    answer: Optional[str]
    sources: Sequence[AnswerSource[ID]]
    search_results: Sequence[SearchResult[ID]]


class MultipleChunkRetrieverQa(
    Task[RetrieverBasedQaInput, MultipleChunkRetrieverQaOutput[ID]], Generic[ID]
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
    """

    def __init__(
        self,
        retriever: BaseRetriever[ID],
        model: ControlModel | None = None,
        insert_chunk_number: int = 5,
        insert_chunk_size: int = 256,
        single_chunk_qa: Task[SingleChunkQaInput, SingleChunkQaOutput] | None = None,
    ):
        super().__init__()
        self._model = model or LuminousControlModel("luminous-supreme-control")
        self._search = Search(retriever)
        self._expand_chunks = ExpandChunks(retriever, self._model, insert_chunk_size)
        self._single_chunk_qa = single_chunk_qa or SingleChunkQa(self._model)
        self._insert_chunk_number = insert_chunk_number

    @staticmethod
    def _combine_input_texts(chunks: Sequence[str]) -> tuple[TextChunk, Sequence[int]]:
        start_indices: list[int] = []
        combined_text = ""
        for chunk in chunks:
            start_indices.append(len(combined_text))
            combined_text += chunk.strip() + "\n\n"
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
                        end=(
                            highlight.end - current_start
                            if isinstance(next_start, float)
                            else min(next_start, highlight.end - current_start)
                        ),
                        score=highlight.score,
                    )
                    current_overlaps.append(highlights_with_indices_fixed)

            overlapping_ranges.append(current_overlaps)
        return overlapping_ranges

    def _expand_search_result_chunks(
        self, search_results: Sequence[SearchResult[ID]], task_span: TaskSpan
    ) -> Sequence[EnrichedChunk[ID]]:

        chunks_to_insert: list[EnrichedChunk[ID]] = []
        for result in search_results:
            input = ExpandChunksInput(
                document_id=result.id, chunks_found=[result.document_chunk]
            )
            expand_chunks_output = self._expand_chunks.run(input, task_span)
            for chunk in expand_chunks_output.chunks:
                if len(chunks_to_insert) >= self._insert_chunk_number:
                    break

                enriched_chunk = EnrichedChunk(
                    document_id=result.id,
                    chunk=chunk.chunk,
                    indices=(chunk.start_index, chunk.end_index),
                )

                if enriched_chunk in chunks_to_insert:
                    continue

                chunks_to_insert.append(enriched_chunk)

        return chunks_to_insert

    def do_run(
        self, input: RetrieverBasedQaInput, task_span: TaskSpan
    ) -> MultipleChunkRetrieverQaOutput[ID]:
        search_output = self._search.run(
            SearchInput(query=input.question), task_span
        ).results
        sorted_search_results = sorted(
            search_output, key=lambda output: output.score, reverse=True
        )

        chunks_to_insert = self._expand_search_result_chunks(
            sorted_search_results, task_span
        )

        chunk_for_prompt, chunk_start_indices = self._combine_input_texts(
            [c.chunk for c in chunks_to_insert]
        )

        single_chunk_qa_input = SingleChunkQaInput(
            chunk=chunk_for_prompt,
            question=input.question,
            language=input.language,
        )

        single_chunk_qa_output = self._single_chunk_qa.run(
            single_chunk_qa_input, task_span
        )

        highlights_per_chunk = self._get_highlights_per_chunk(
            chunk_start_indices, single_chunk_qa_output.highlights
        )

        return MultipleChunkRetrieverQaOutput(
            answer=single_chunk_qa_output.answer,
            sources=[
                AnswerSource(
                    chunk=enriched_chunk,
                    highlights=highlights,
                )
                for enriched_chunk, highlights in zip(
                    chunks_to_insert, highlights_per_chunk
                )
            ],
            search_results=sorted_search_results,
        )
