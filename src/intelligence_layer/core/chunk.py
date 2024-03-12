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
