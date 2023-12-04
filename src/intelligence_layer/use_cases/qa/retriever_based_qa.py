from pydantic import BaseModel

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.connectors.retrievers.base_retriever import BaseRetriever
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan
from intelligence_layer.use_cases.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.use_cases.search.search import Search, SearchInput


class RetrieverBasedQaInput(BaseModel):
    """The input for a `RetrieverBasedQa` task.

    Attributes:
        question: The question to be answered based on the documents accessed
            by the retriever.
        language: The desired language of the answer. ISO 619 str with language e.g. en, fr, etc.
    """

    question: str
    language: Language = Language("en")


class RetrieverBasedQa(Task[RetrieverBasedQaInput, MultipleChunkQaOutput]):
    """Answer a question based on documents found by a retriever.

    `RetrieverBasedQa` is a task that answers a question based on a set of documents.
    Relies on some retriever of type `BaseRetriever` that has the ability to access texts.

    Note:
        `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        retriever: Used to access and return a set of texts.
        model: A valid Aleph Alpha model name.
        allowed_languages: List of languages to which the language detection is limited (ISO619).
        fallback_language: The default language of the output.

    Example:
        >>> import os
        >>> from intelligence_layer.connectors import DocumentIndexClient
        >>> from intelligence_layer.connectors import LimitedConcurrencyClient
        >>> from intelligence_layer.connectors import DocumentIndexRetriever
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.use_cases import RetrieverBasedQa, RetrieverBasedQaInput


        >>> token = os.getenv("AA_TOKEN")
        >>> client = LimitedConcurrencyClient.from_token(token)
        >>> document_index = DocumentIndexClient(token)
        >>> retriever = DocumentIndexRetriever(document_index, "aleph-alpha", "wikipedia-de", 3)
        >>> task = RetrieverBasedQa(client, retriever)
        >>> input_data = RetrieverBasedQaInput(question="When was Rome founded?")
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input_data, tracer)
    """

    def __init__(
        self,
        client: AlephAlphaClientProtocol,
        retriever: BaseRetriever,
        model: str = "luminous-supreme-control",
    ):
        super().__init__()
        self._client = client
        self._model = model
        self._search = Search(retriever)
        self._multi_chunk_qa = MultipleChunkQa(self._client, self._model)

    def do_run(
        self, input: RetrieverBasedQaInput, task_span: TaskSpan
    ) -> MultipleChunkQaOutput:
        search_output = self._search.run(SearchInput(query=input.question), task_span)

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[Chunk(result.document.text) for result in search_output.results],
            question=input.question,
            language=input.language,
        )
        return self._multi_chunk_qa.run(multi_chunk_qa_input, task_span)
