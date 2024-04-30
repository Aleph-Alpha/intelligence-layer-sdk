from functools import lru_cache
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
        max_chunk_size: The maximum chunk size of each returned chunk.
    """

    def __init__(
        self,
        retriever: BaseRetriever[ID],
        model: AlephAlphaModel,
        max_chunk_size: int = 512,
    ):
        super().__init__()
        self._retriever = retriever
        self._chunk_with_indices = ChunkWithIndices(model, max_chunk_size)
        self._no_op_tracer = NoOpTracer()

    def do_run(
        self, input: ExpandChunksInput[ID], task_span: TaskSpan
    ) -> ExpandChunksOutput:
        chunked_text = self._retrieve_and_chunk(input.document_id)

        overlapping_chunk_indices = self._overlapping_chunk_indices(
            [(c.start_index, c.end_index) for c in chunked_text],
            [(chunk.start, chunk.end) for chunk in input.chunks_found],
        )

        return ExpandChunksOutput(
            chunks=[chunked_text[index] for index in overlapping_chunk_indices],
        )

    @lru_cache(maxsize=100)
    def _retrieve_and_chunk(
        self, document_id: ID
    ) -> Sequence[ChunkWithStartEndIndices]:
        text = self._retrieve_text(document_id)
        return self._chunk_text(text)

    def _retrieve_text(self, document_id: ID) -> str:
        full_document = self._retriever.get_full_document(document_id)
        if not full_document:
            raise RuntimeError(f"No document for id '{document_id}' found")
        return full_document.text

    def _chunk_text(self, text: str) -> Sequence[ChunkWithStartEndIndices]:
        # NoOpTracer used to allow caching {ID: Sequence[ChunkWithStartEndIndices]}
        return self._chunk_with_indices.run(
            ChunkInput(text=text), self._no_op_tracer
        ).chunks_with_indices

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
