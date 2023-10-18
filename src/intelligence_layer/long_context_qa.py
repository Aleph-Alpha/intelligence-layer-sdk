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
    """This is the input for QA method working on long contex texts 

    Attributes:
        text: Text of an arbitrary length on the basis of which the question is to be answered.
        question: The question for the text.         
    """
    text: str
    question: str


class SearchResult(BaseModel):
    """For document retrieval we return a chunk alongside its similarity score

    Attributes:
        score: The similarity score between the document and the query.
        chunk: A segment of the orginal long text that is relevant to the query.               
    """
    score: float
    chunk: str

class LongContextQa(Task[LongContextQaInput, MultipleChunkQaOutput]):
    """Answer question for lengthy documents 

    LongContextQa is a task answering a question for a long document, where the length
    of text exceeds the context length of a model (e.g. 2048 tokens for the luminous models).

    Args:
        client: An instance of the Aleph Alpha client.
        max_tokens_in_chunk: Maximum number of tokens in each chunk.
        k: The number of top relevant chunks to retrieve.
        model: Identifier of the model to be used.

    Example:
        >>> client = Client(token=os.getenv("AA_TOKEN"))
        >>> task = LongContextQa(client)
        >>> input_data = LongContextQaInput(text="Lengthy text goes here...", question="What is the main point?")
        >>> logger = InMemoryDebugLogger(name="qa")
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
        """
        The main question-answering method.
        It first splits the text into n smaller chunks, then we use the
        text embeddings to find k chunks that are most similar to a given question,
        and we feed them into 'MultipleChunkQa', which handles question answering.

        For splitting we use the semantic_text_splitter that consumes the tokenizer and
        prevents the chunk from being too long (having too many tokens).

        What it does under the hood is answer questions about each individual chunk, and
        feeds the individual answers into a prompt combining multiple answers.
        After that, this final answer is added to the output."""

        chunks = self._splitter.chunks(input.text, self._max_tokens_in_chunk)
        logger.log("chunks", chunks)

        retriever = InMemoryRetriever(self._client, chunks=chunks, threshold=0.5)

        relevant_chunks_with_scores = retriever.get_relevant_documents_with_scores(
            input.question, k=self._k, logger=logger.child_logger("Retriever")
        )

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[result.chunk for result in relevant_chunks_with_scores],
            question=input.question,
        )

        qa_output = self._multi_chunk_qa.run(
            multi_chunk_qa_input, logger.child_logger("Multi Chunk QA")
        )

        return qa_output
