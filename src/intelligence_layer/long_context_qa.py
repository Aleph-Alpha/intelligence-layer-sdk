from typing import Optional, Sequence
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from aleph_alpha_client import (
    Client,
    Prompt, SemanticRepresentation, SemanticEmbeddingRequest
)
from pydantic import BaseModel
from intelligence_layer.multiple_chunk_qa import MultipleChunkQa, MultipleChunkQaInput, MultipleChunkQaOutput
from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from intelligence_layer.prompt_template import (
    PromptRange,
    PromptTemplate,
    PromptWithMetadata,
    TextCursor,
)
from intelligence_layer.task import DebugLog, LogLevel, Task
from semantic_text_splitter import HuggingFaceTextSplitter


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
        log_level: LogLevel,
        max_tokens_in_chunk: int = 512,
        k: int = 4,
        threshold: float = 0.5,
        model: str = "luminous-supreme-control"
    ):
        self.client = client
        self.search_client = QdrantClient(":memory:")
        self.log_level = log_level
        self.model = model
        self.max_tokens_in_chunk = max_tokens_in_chunk
        self.tokenizer = self.client.tokenizer(model)
        self.splitter = HuggingFaceTextSplitter(self.tokenizer, trim_chunks=True)
        self.multi_chunk_qa = MultipleChunkQa(self.client, self.log_level, self.model)
        self.k = k
        self.threshold = threshold

    def run(self, input: LongContextQaInput) -> MultipleChunkQaOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        chunks = self.splitter.chunks(input.text, self.max_tokens_in_chunk)
        relevant_chunks_with_scores = self.find_relevant_chunks(chunks=chunks, question=input.question)


        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[result.chunk for result in relevant_chunks_with_scores], 
            question=input.question
            )
        
        # should we add something from the chunking to debug log? this debug log is duplicate for now
        debug_log.info("Input", {"question": input.question, "text": input.text})
        qa_output = self.multi_chunk_qa.run(multi_chunk_qa_input)
        debug_log.debug("Multi Chunk QA", qa_output.debug_log)
        
        qa_output.debug_log = debug_log
        
        return qa_output

    def find_relevant_chunks(self, chunks: Sequence[str], question: str) -> Sequence[SearchResult]:
        def _point_to_search_result(point: ScoredPoint) -> SearchResult:
            assert point.payload
            return SearchResult(score=point.score, chunk=point.payload["text"])
        
        chunk_embeddings = []
        for chunk in chunks:
            chunk_embedding_request = SemanticEmbeddingRequest(
                prompt = Prompt.from_text(chunk),
                representation= SemanticRepresentation.Document,
                compress_to_size= 128,
                normalize= True
            )
            chunk_embedding_response = self.client.semantic_embed(request=chunk_embedding_request, model="luminous-base")
            chunk_embeddings.append(chunk_embedding_response.embedding)

        question_embedding_request = SemanticEmbeddingRequest(
            prompt = Prompt.from_text(question),
            representation = SemanticRepresentation.Query,
            compress_to_size = 128,
            normalize = True
        )
        question_embedding = self.client.semantic_embed(request=question_embedding_request, model="luminous-base").embedding

        self.search_client.recreate_collection(
            collection_name="lokal_chunks",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )
        operation_info =self.search_client.upsert(
                collection_name="lokal_chunks",
                wait=True,
                points=[
                    PointStruct(id=idx, vector=chunk_embedding, payload={"text": chunk}) for idx, (chunk_embedding, chunk) in enumerate(zip(chunk_embeddings, chunks))
                ]
        )
        search_result =self.search_client.search(
            collection_name="lokal_chunks",
            query_vector=question_embedding,
            score_threshold=self.threshold,
            limit=self.k
        )

        output = [_point_to_search_result(point) for point in search_result]
        return output



