from typing import Sequence
from aleph_alpha_client import (
    Client,
)
from pydantic import BaseModel
from intelligence_layer.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.retrievers.base import BaseRetriever
from intelligence_layer.search import Search, SearchInput
from intelligence_layer.task import Chunk, DebugLogger, Task
from semantic_text_splitter import HuggingFaceTextSplitter
from intelligence_layer.retrievers.in_memory import InMemoryRetriever


class RetrieverBasedQaInput(BaseModel):
    """This is the input for a retriever based QA task.

    Attributes:
        question: The question to be answered against the retriever.
    """

    question: str


class RetrieverBasedQa(Task[RetrieverBasedQaInput, MultipleChunkQaOutput]):
    """Answer a question against documents in a retriever.

    `RetrieverBasedQa` is a task that answers a question based on a set of documents.
    These documents are served in form of a `BaseRetriever`.

    Args:
        client: An instance of the Aleph Alpha client
        retriever: Used to access the documents
        model: Identifier of the model to be used

    Example:
        >>> client = Client(token=os.getenv("AA_TOKEN"))
        >>> task = RetrieverBasedQa(client)
        >>> input_data = RetrieverBasedQaInput(question="What is the main point?")
        >>> logger = InMemoryDebugLogger(name="qa")
        >>> output = task.run(input_data, logger)
        >>> print(output.answer)
    """

    def __init__(
        self,
        client: Client,
        retriever: BaseRetriever,
        model: str = "luminous-supreme-control",
    ):
        self._client = client
        self._model = model
        self._search = Search(retriever)
        self._multi_chunk_qa = MultipleChunkQa(self._client, self._model)

    def run(
        self, input: RetrieverBasedQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        search_output = self._search.run(SearchInput(query=input.question), logger)
        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[Chunk(result.text) for result in search_output.results],
            question=input.question,
        )
        return self._multi_chunk_qa.run(multi_chunk_qa_input, logger)
