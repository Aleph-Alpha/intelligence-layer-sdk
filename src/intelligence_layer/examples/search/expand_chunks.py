from collections import OrderedDict
from typing import Generic, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors import BaseRetriever, DocumentChunk
from intelligence_layer.connectors.retrievers.base_retriever import ID
from intelligence_layer.core import (
    ChunkInput,
    ChunkWithIndices,
    ChunkWithStartEndIndices,
    NoOpTracer,
)
from intelligence_layer.core.model import AlephAlphaModel
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer.tracer import TaskSpan


class ExpandChunksInput(BaseModel, Generic[ID]):
    document_id: ID
    chunks_found: Sequence[DocumentChunk]


class ExpandChunksOutput(BaseModel):
    chunks: Sequence[ChunkWithStartEndIndices]


class ExpandChunks(Generic[ID], Task[ExpandChunksInput[ID], ExpandChunksOutput]):
    """Expand chunks found during search.

    Args:
        retriever: Used to access and return a set of texts.
        model: The model's tokenizer is relevant to calculate the correct size of the returned chunks.
        max_chunk_size: The maximum chunk size of each returned chunk in #tokens.
    """

    def __init__(
        self,
        retriever: BaseRetriever[ID],
        model: AlephAlphaModel,
        max_chunk_size: int = 512,
    ):
        super().__init__()
        self._retriever = retriever
        self._target_chunker = ChunkWithIndices(model, max_chunk_size)
        self._large_chunker = ChunkWithIndices(model, max_chunk_size * 32)
        self._no_op_tracer = NoOpTracer()

    def do_run(
        self, input: ExpandChunksInput[ID], task_span: TaskSpan
    ) -> ExpandChunksOutput:
        text = self._retrieve_text(input.document_id)
        large_chunks = self._expand_chunks(
            text, 0, input.chunks_found, self._large_chunker
        )
        nested_expanded_chunks = [
            self._expand_chunks(
                chunk.chunk, chunk.start_index, input.chunks_found, self._target_chunker
            )
            for chunk in large_chunks
        ]
        return ExpandChunksOutput(
            # deduplicating while preserving order
            chunks=list(
                OrderedDict.fromkeys(
                    chunk
                    for large_chunk in nested_expanded_chunks
                    for chunk in large_chunk
                )
            )
        )

    def _retrieve_text(self, document_id: ID) -> str:
        full_document = self._retriever.get_full_document(document_id)
        if not full_document:
            raise RuntimeError(f"No document for id '{document_id}' found")
        return full_document.text

    def _expand_chunks(
        self,
        text: str,
        text_start: int,
        chunks_found: Sequence[DocumentChunk],
        chunker: ChunkWithIndices,
    ) -> Sequence[ChunkWithStartEndIndices]:
        chunked_text = self._chunk_text(text, chunker)

        overlapping_chunk_indices = self._overlapping_chunk_indices(
            [
                (c.start_index + text_start, c.end_index + text_start)
                for c in chunked_text
            ],
            [(chunk.start, chunk.end) for chunk in chunks_found],
        )

        return [chunked_text[index] for index in overlapping_chunk_indices]

    def _chunk_text(
        self, text: str, chunker: ChunkWithIndices
    ) -> Sequence[ChunkWithStartEndIndices]:
        chunks = chunker.run(
            ChunkInput(text=text), self._no_op_tracer
        ).chunks_with_indices
        return chunks

    def _overlapping_chunk_indices(
        self,
        chunk_indices: Sequence[tuple[int, int]],
        target_ranges: Sequence[tuple[int, int]],
    ) -> list[int]:
        overlapping_indices: list[int] = []

        for i in range(len(chunk_indices)):
            if any(
                (
                    chunk_indices[i][0] <= target_range[1]
                    and chunk_indices[i][1] > target_range[0]
                )
                for target_range in target_ranges
            ):
                overlapping_indices.append(i)

        return overlapping_indices
