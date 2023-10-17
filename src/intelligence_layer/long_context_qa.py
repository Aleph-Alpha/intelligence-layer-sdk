from aleph_alpha_client import (
    Client,
)
from pydantic import BaseModel
from intelligence_layer.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.task import DebugLogger, Task
from semantic_text_splitter import HuggingFaceTextSplitter
from intelligence_layer.retrievers.in_memory_retriever import InMemoryRetriever


class LongContextQaInput(BaseModel):
    text: str
    question: str


class SearchResult(BaseModel):
    score: float
    chunk: str


NO_ANSWER_TEXT = "NO_ANSWER_IN_TEXT"


class LongContextQa(Task[LongContextQaInput, MultipleChunkQaOutput]):
    """
    LongContextQa is a task answering a question for a long document, where the length
    of text exceeds the context length of a model (e.g. 2048 tokens for the luminous models).


    Attributes:
        NO_ANSWER_TEXT: A constant representing no answer in the context.

    Args:
        client: An instance of the Aleph Alpha client.
        max_tokens_in_chunk: Maximum number of tokens in each chunk.
        k: The number of top relevant chunks to retrieve.
        model: Identifier of the model to be used.

    Methods:
        run(): The main question-answering method It first splits the text.
                into smaller chunks, then we use the text embeddings to find chunks that are most similar.
                to a given question, and we feed them into 'MultipleChunkQa', which handles question answering.

                What it does under the hood is answer questions about each individual chunk, and
                feeds the individual answers into a prompt combining multiple answers.
                After that, this final answer is added to the output.

    Example:
        >>> client = Client(api_key="YOUR_API_KEY")
        >>> task = LongContextQa(client)
        >>> input_data = LongContextQaInput(text="Lengthy text goes here...", question="What is the main point?")
        >>> logger = DebugLogger()
        >>> output = task.run(input_data, logger)
        >>> print(output.answer)
    """

    def __init__(
        self,
        client: Client,
        max_tokens_in_chunk: int = 512,
        k: int = 4,
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.model = model
        self.max_tokens_in_chunk = max_tokens_in_chunk
        self.tokenizer = self.client.tokenizer(model)
        self.splitter = HuggingFaceTextSplitter(self.tokenizer, trim_chunks=True)
        self.multi_chunk_qa = MultipleChunkQa(self.client, self.model)
        self.k = k

    def run(
        self, input: LongContextQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        chunks = self.splitter.chunks(input.text, self.max_tokens_in_chunk)
        logger.log("chunks", chunks)

        retriever = InMemoryRetriever(self.client, chunks=chunks, threshold=0.5)

        relevant_chunks_with_scores = retriever.get_relevant_documents_with_scores(
            input.question, k=self.k, logger=logger.child_logger("Retriever")
        )

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[result.chunk for result in relevant_chunks_with_scores],
            question=input.question,
        )

        qa_output = self.multi_chunk_qa.run(
            multi_chunk_qa_input, logger.child_logger("Multi Chunk QA")
        )

        return qa_output
