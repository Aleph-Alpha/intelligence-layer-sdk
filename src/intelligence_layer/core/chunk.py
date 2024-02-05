from typing import NewType, Sequence

from pydantic import BaseModel
from semantic_text_splitter import HuggingFaceTextSplitter

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan

Chunk = NewType("Chunk", str)
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

    chunks: Sequence[Chunk]


class ChunkTask(Task[ChunkInput, ChunkOutput]):
    """Splits a longer text into smaller text chunks.

    Provide a text of any length and chunk it into smaller pieces using a
    tokenizer that is available within the Aleph Alpha client.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.
        max_tokens_per_chunk: The maximum number of tokens to fit into one chunk.
    """

    def __init__(
        self, client: AlephAlphaClientProtocol, model: str, max_tokens_per_chunk: int
    ):
        super().__init__()
        tokenizer = client.tokenizer(model)
        self._splitter = HuggingFaceTextSplitter(tokenizer)
        self._max_tokens_per_chunk = max_tokens_per_chunk

    def do_run(self, input: ChunkInput, task_span: TaskSpan) -> ChunkOutput:
        chunks = [
            Chunk(t)
            for t in self._splitter.chunks(input.text, self._max_tokens_per_chunk)
        ]
        return ChunkOutput(chunks=chunks)


class ChunkOverlapTask(Task[ChunkInput, ChunkOutput]):
    """Splits a longer text into smaller text chunks, where every chunk overlaps
    with the previous chunk by `overlap_length_tokens` number of tokens.

    Provide a text of any length and chunk it into smaller pieces using a
    tokenizer that is available within the Aleph Alpha client.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.
        max_tokens_per_chunk: The maximum number of tokens to fit into one chunk.
        overlap_length_tokens: The number of tokens every chunk overlaps with the previous chunk.
    """

    def __init__(
        self,
        client: AlephAlphaClientProtocol,
        model: str,
        max_tokens_per_chunk: int,
        overlap_length_tokens: int
    ):
        super().__init__()
        self.chunk_task = ChunkTask(client, model, max_tokens_per_chunk-overlap_length_tokens)
        self.tokenizer = client.tokenizer(model)
        self.overlap_length_tokens = overlap_length_tokens

    def do_run(self, input: ChunkInput, task_span: TaskSpan) -> ChunkOutput:
        chunks = self.chunk_task.run(input, task_span).chunks
        token_chunks = self.tokenizer.encode_batch(chunks)
        chunk_tokens = [token_chunks[0].tokens] + [
            token_chunks[i].tokens[-self.overlap_length_tokens:].append(token_chunks[i+1].tokens)
            for i in range(len(token_chunks)-1)
        ]
        decoded_chunks = self.tokenizer.decode_batch(chunk_tokens)
        return ChunkOutput(chunks=decoded_chunks)

