from aleph_alpha_client import (
    Client,
)
from pydantic import BaseModel
from intelligence_layer.core.detect_language import (
    DetectLanguage,
    DetectLanguageInput,
    Language,
)
from intelligence_layer.core.detect_language import (
    DetectLanguage,
    DetectLanguageInput,
    Language,
)

from intelligence_layer.use_cases.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.connectors.retrievers.base_retriever import BaseRetriever
from intelligence_layer.use_cases.search.search import Search, SearchInput
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.task import Task
from intelligence_layer.core.logger import DebugLogger


class RetrieverBasedQaInput(BaseModel):
    """The input for a `RetrieverBasedQa` task.

    Attributes:
        question: The question to be answered based on the documents accessed
            by the retriever.
    """

    question: str

class RetrieverBasedQa(Task[RetrieverBasedQaInput, MultipleChunkQaOutput]):
    """Answer a question based on documents found by a retriever.

    RetrieverBasedQa` is a task that answers a question based on a set of documents.
    Relies on some retriever of type `BaseRetriever` that has the ability to access texts.

    Note:
        `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        retriever: Used to access and return a set of texts.
        model: A valid Aleph Alpha model name.

    Example:
        >>> token = os.getenv("AA_TOKEN")
        >>> client = Client(token)
        >>> document_index = DocumentIndex(token)
        >>> retriever = DocumentIndexRetriever(document_index, "my_namespace", "ancient_facts_collection", 3)
        >>> task = RetrieverBasedQa(client, retriever)
        >>> input_data = RetrieverBasedQaInput(question="When was Rome founded?")
        >>> logger = InMemoryDebugLogger(name="Retriever Based QA")
        >>> output = task.run(input_data, logger)
        >>> print(output.answer)
        Rome was founded in 753 BC.
    """

    def __init__(
        self,
        client: Client,
        retriever: BaseRetriever,
        model: str = "luminous-supreme-control",
    ):
        super().__init__()
        self._client = client
        self._model = model
        self._search = Search(retriever)
        self._multi_chunk_qa = MultipleChunkQa(self._client, self._model)
        self._language_detector = DetectLanguage(threshold=0.5)
        self._fallback_language = Language("en")
        self._language_detector = DetectLanguage(threshold=0.5)
        self._fallback_language = Language("en")

    def run(
        self, input: RetrieverBasedQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        search_output = self._search.run(SearchInput(query=input.question), logger)

        question_language = (
            self._language_detector.run(
                DetectLanguageInput(text=input.question), logger
            ).best_fit
            or self._fallback_language
        )


        question_language = (
            self._language_detector.run(
                DetectLanguageInput(text=input.question), logger
            ).best_fit
            or self._fallback_language
        )

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[Chunk(result.document.text) for result in search_output.results],
            question=input.question,
            language=question_language,
        )
        return self._multi_chunk_qa.run(multi_chunk_qa_input, logger)
