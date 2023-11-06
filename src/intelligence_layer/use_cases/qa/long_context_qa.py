from typing import Sequence

from aleph_alpha_client import Client
from pydantic import BaseModel
from semantic_text_splitter import HuggingFaceTextSplitter
from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.core.detect_language import Language

from intelligence_layer.use_cases.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.use_cases.search.search import Search, SearchInput
from intelligence_layer.core.task import Chunk, Task
from intelligence_layer.core.logger import DebugLogger


class LongContextQaInput(BaseModel):
    """The input for a `LongContextQa` task.

    Attributes:
        text: Text of arbitrary length on the basis of which the question is to be answered.
        question: The question for the text.
    """

    text: str
    question: str
    language: Language = Language("en")


class LongContextQa(Task[LongContextQaInput, MultipleChunkQaOutput]):
    """Answer a question on the basis of a (lengthy) document.

    Best for answering a question on the basis of a long document, where the length
    of text exceeds the context length of a model (e.g. 2048 tokens for the luminous models).

    Note:
        - Creates instance of `InMemoryRetriever` on the fly.
        - `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        max_tokens_in_chunk: The input text will be split into chunks to fit the context window.
            Used to tweak the length of the chunks.
        k: The number of top relevant chunks to retrieve.
        model: A valid Aleph Alpha model name.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = LongContextQa(client)
        >>> input = LongContextQaInput(text="Lengthy text goes here...", question="Where does the text go?")
        >>> logger = InMemoryDebugLogger(name="Long Context QA")
        >>> output = task.run(input, logger)
    """

    def __init__(
        self,
        client: Client,
        max_tokens_in_chunk: int = 512,
        k: int = 4,
        model: str = "luminous-supreme-control",
    ):
        super().__init__()
        self._client = client
        self._model = model
        self._max_tokens_in_chunk = max_tokens_in_chunk
        self._tokenizer = self._client.tokenizer(model)
        self._splitter = HuggingFaceTextSplitter(self._tokenizer, trim_chunks=True)
        self._multi_chunk_qa = MultipleChunkQa(self._client, self._model)
        self._k = k

    def run(
        self, input: LongContextQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        chunks = self._chunk(input.text)
        logger.log("chunks", chunks)
        retriever = QdrantInMemoryRetriever(
            self._client,
            documents=[Document(text=c) for c in chunks],
            k=self._k,
            threshold=0.5,
        )
        search_output = Search(retriever).run(SearchInput(query=input.question), logger)
        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[Chunk(result.document.text) for result in search_output.results],
            question=input.question,
            language=input.language,
        )
        qa_output = self._multi_chunk_qa.run(multi_chunk_qa_input, logger)
        return qa_output

    def _chunk(self, text: str) -> Sequence[Chunk]:
        return [
            Chunk(t) for t in self._splitter.chunks(text, self._max_tokens_in_chunk)
        ]
