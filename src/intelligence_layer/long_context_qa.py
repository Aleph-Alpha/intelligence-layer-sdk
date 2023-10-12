from typing import Optional
from aleph_alpha_client import (
    Client,
)
from pydantic import BaseModel
from intelligence_layer.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.retrievers.base import BaseRetriver
from intelligence_layer.task import DebugLogger, Task, log_run_input_output
from semantic_text_splitter import HuggingFaceTextSplitter
from intelligence_layer.retrievers.qdrant import QdrantRetriever


class LongContextQaInput(BaseModel):
    text: str
    question: str


class SearchResult(BaseModel):
    score: float
    chunk: str


NO_ANSWER_TEXT = "NO_ANSWER_IN_TEXT"


class LongContextQa(Task[LongContextQaInput, MultipleChunkQaOutput]):
    def __init__(
        self,
        client: Client,
        max_tokens_in_chunk: int = 512,
        k: int = 4,
        retriever: Optional[BaseRetriver] = None,
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.model = model
        self.max_tokens_in_chunk = max_tokens_in_chunk
        self.tokenizer = self.client.tokenizer(model)
        self.splitter = HuggingFaceTextSplitter(self.tokenizer, trim_chunks=True)
        self.multi_chunk_qa = MultipleChunkQa(self.client, self.model)
        self.k = k

        self.qdrant_retriever = retriever or QdrantRetriever(client, threshold=0.5)

    @log_run_input_output
    def run(
        self, input: LongContextQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        chunks = self.splitter.chunks(input.text, self.max_tokens_in_chunk)

        self.qdrant_retriever.add_documents(chunks)

        relevant_chunks_with_scores = (
            self.qdrant_retriever.get_relevant_documents_with_scores(
                input.question, k=self.k
            )
        )
        self.qdrant_retriever.clear()

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[result.chunk for result in relevant_chunks_with_scores],
            question=input.question,
        )

        qa_output = self.multi_chunk_qa.run(
            multi_chunk_qa_input, logger.child_logger("Multi Chunk QA")
        )

        return qa_output
