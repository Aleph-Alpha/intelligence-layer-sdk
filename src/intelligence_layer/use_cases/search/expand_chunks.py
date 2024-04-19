from typing import Generic, Sequence

from pydantic import BaseModel

from intelligence_layer.connectors import BaseRetriever, DocumentChunk
from intelligence_layer.connectors.retrievers.base_retriever import ID
from intelligence_layer.core.chunk import (
    ChunkInput,
    ChunkWithIndices,
    ChunkWithStartEndIndices,
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

    def do_run(
        self, input: ExpandChunksInput[ID], task_span: TaskSpan
    ) -> ExpandChunksOutput:
        full_doc = self._retriever.get_full_document(input.document_id)
        if not full_doc:
            raise RuntimeError(f"No document for id '{input.document_id}' found")

        chunk_with_indices = self._chunk_with_indices.run(
            ChunkInput(text=full_doc.text), task_span
        ).chunks_with_indices

        overlapping_chunk_indices = self._overlapping_chunk_indices(
            [(c.start_index, c.end_index) for c in chunk_with_indices],
            [(chunk.start, chunk.end) for chunk in input.chunks_found],
        )

        return ExpandChunksOutput(
            chunks=[chunk_with_indices[index] for index in overlapping_chunk_indices],
        )

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
