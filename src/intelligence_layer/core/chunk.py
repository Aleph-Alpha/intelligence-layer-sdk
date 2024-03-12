from typing import NewType, Sequence

from pydantic import BaseModel
from semantic_text_splitter import TextSplitter

from intelligence_layer.core.model import AlephAlphaModel
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer.tracer import TaskSpan

TextChunk = NewType("TextChunk", str)
"""Segment of a larger text.

This type infers that the string is smaller than the context size of the model where it is used.

LLMs can't process documents larger than their context size.
To handle this, documents have to be split up into smaller segments that fit within their context size.
These smaller segments are referred to as chunks.
"""


class ChunkInput(BaseModel):
    """The input for a `ChunkTask`.

    Attributes:
        text: A text of arbitrary length.
    """

    text: str


class ChunkOutput(BaseModel):
    """The output of a `ChunkTask`.

    Attributes:
        chunks: A list of smaller sections of the input text.
    """

    chunks: Sequence[TextChunk]


class Chunk(Task[ChunkInput, ChunkOutput]):
    """Splits a longer text into smaller text chunks.

    Provide a text of any length and chunk it into smaller pieces using a
    tokenizer that is available within the Aleph Alpha client.

    Args:
        model: A valid Aleph Alpha model.
        max_tokens_per_chunk: The maximum number of tokens to fit into one chunk.
    """

    def __init__(self, model: AlephAlphaModel, max_tokens_per_chunk: int = 512):
        super().__init__()
        self._splitter = TextSplitter.from_huggingface_tokenizer(model.get_tokenizer())
        self._max_tokens_per_chunk = max_tokens_per_chunk

    def do_run(self, input: ChunkInput, task_span: TaskSpan) -> ChunkOutput:
        chunks = [
            TextChunk(t)
            for t in self._splitter.chunks(input.text, self._max_tokens_per_chunk)
        ]
        return ChunkOutput(chunks=chunks)


class ChunkOverlap(Task[ChunkInput, ChunkOutput]):
    """Splits a longer text into smaller text chunks, where every chunk overlaps
    with the previous chunk by `overlap_length_tokens` number of tokens.

    Provide a text of any length and chunk it into smaller pieces using a
    tokenizer that is available within the Aleph Alpha client.

    Args:
        model: A valid Aleph Alpha model.
        max_tokens_per_chunk: The maximum number of tokens to fit into one chunk.
        overlap_length_tokens: The number of tokens every chunk overlaps with the previous chunk.
    """

    def __init__(
        self,
        model: AlephAlphaModel,
        max_tokens_per_chunk: int = 512,
        overlap_length_tokens: int = 0,
    ):
        super().__init__()
        if overlap_length_tokens >= max_tokens_per_chunk:
            raise RuntimeError(
                "Cannot choose an overlap ({}) longer than the chunk ({})".format(
                    overlap_length_tokens, max_tokens_per_chunk
                )
            )

        self.tokenizer = model.get_tokenizer()
        self.chunk_task = Chunk(model, overlap_length_tokens // 2)
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_length_tokens = overlap_length_tokens

    def do_run(self, input: ChunkInput, task_span: TaskSpan) -> ChunkOutput:
        chunks = self.chunk_task.run(input, task_span).chunks
        id_chunks = self.tokenizer.encode_batch(chunks)

        chunk_ids = [id_chunks[0].ids]
        current_chunk = chunk_ids[0]
        last_overlap = [chunk_ids[0]]
        for chunk in id_chunks[1:]:
            if len(chunk.ids) + len(current_chunk) <= self.max_tokens_per_chunk:
                current_chunk.extend(chunk.ids)
            else:
                current_chunk = sum(last_overlap, []) + chunk.ids
                chunk_ids.append(current_chunk)

            last_overlap.append(chunk.ids)
            total_length = len(sum(last_overlap, []))
            while total_length > self.overlap_length_tokens:
                total_length -= len(last_overlap[0])
                last_overlap = last_overlap[1:]

        decoded_chunks = self.tokenizer.decode_batch(chunk_ids)
        return ChunkOutput(chunks=decoded_chunks)
