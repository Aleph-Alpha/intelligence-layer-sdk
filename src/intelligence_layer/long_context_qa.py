from typing import Optional, Sequence
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticRepresentation,
    SemanticEmbeddingRequest,
)
from pydantic import BaseModel
from intelligence_layer.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.retrivers.base import BaseRetriver
from intelligence_layer.task import DebugLogger, Task, log_run_input_output
from semantic_text_splitter import HuggingFaceTextSplitter
from intelligence_layer.retrivers.qdrant import QdrantRetriver


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
        retriver: Optional[BaseRetriver] = None,
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.model = model
        self.max_tokens_in_chunk = max_tokens_in_chunk
        self.tokenizer = self.client.tokenizer(model)
        self.splitter = HuggingFaceTextSplitter(self.tokenizer, trim_chunks=True)
        self.multi_chunk_qa = MultipleChunkQa(self.client, self.model)

        self.k = k

        self.qdrant_retriver = retriver or QdrantRetriver(client, threshold=0.5)

    @log_run_input_output
    def run(
        self, input: LongContextQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        chunks = self.splitter.chunks(input.text, self.max_tokens_in_chunk)

        self.qdrant_retriver.add_documents(chunks)

        relevant_chunks_with_scores = (
            self.qdrant_retriver.get_relevant_documents_with_scores(
                input.question, k=self.k
            )
        )

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[result.chunk for result in relevant_chunks_with_scores],
            question=input.question,
        )

        # should we add something from the chunking to debug log? this debug log is duplicate for now
        logger.log("Input", {"question": input.question, "text": input.text})
        qa_output = self.multi_chunk_qa.run(
            multi_chunk_qa_input, logger.child_logger("Multi Chunk QA")
        )

        return qa_output
