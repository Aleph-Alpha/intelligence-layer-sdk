from typing import NewType, Sequence
from aleph_alpha_client import Client
from pydantic import BaseModel
from semantic_text_splitter import HuggingFaceTextSplitter
from intelligence_layer.core.logger import DebugLogger

from intelligence_layer.core.task import Task


Chunk = NewType("Chunk", str)
"""Segment of a larger text.

This type infers that the string is smaller than the context size of the model where it is used.

LLMs can't process documents larger than their context size.
To handle this, documents have to be split up into smaller segments that fit within their context size.
These smaller segments are referred to as chunks.
"""


class ChunkInput(BaseModel):
    text: str


class ChunkOutput(BaseModel):
    chunks: Sequence[Chunk]


class ChunkTask(Task[ChunkInput, ChunkOutput]):
    def __init__(self, client: Client, model: str, max_tokens_per_chunk: int):
        super().__init__()
        tokenizer = client.tokenizer(model)
        self._splitter = HuggingFaceTextSplitter(tokenizer)
        self._max_tokens_per_chunk = max_tokens_per_chunk

    def run(self, input: ChunkInput, logger: DebugLogger) -> ChunkOutput:
        chunks = [
            Chunk(t)
            for t in self._splitter.chunks(input.text, self._max_tokens_per_chunk)
        ]
        return ChunkOutput(chunks=chunks)
