from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core import (
    Chunk,
    ChunkInput,
    ChunkTask,
    ControlModel,
    DetectLanguage,
    Language,
    LuminousControlModel,
    Task,
    TaskSpan,
)
from intelligence_layer.use_cases.qa.multiple_chunk_qa import (
    MultipleChunkQa,
    MultipleChunkQaInput,
    MultipleChunkQaOutput,
)
from intelligence_layer.use_cases.search.search import Search, SearchInput


class LongContextQaInput(BaseModel):
    """The input for a `LongContextQa` task.

    Attributes:
        text: Text of arbitrary length on the basis of which the question is to be answered.
        question: The question for the text.
        language: The desired language of the answer. ISO 619 str with language e.g. en, fr, etc.
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
        max_tokens_in_chunk: The input text will be split into chunks to fit the context window.
            Used to tweak the length of the chunks.
        k: The number of top relevant chunks to retrieve.
        model: The model used in the task.

    Example:
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.use_cases import LongContextQa, LongContextQaInput


        >>> task = LongContextQa()
        >>> input = LongContextQaInput(text="Lengthy text goes here...",
        ...                             question="Where does the text go?")
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    def __init__(
        self,
        multi_chunk_qa: Task[MultipleChunkQaInput, MultipleChunkQaOutput] | None = None,
        max_tokens_per_chunk: int = 1024,
        k: int = 4,
        model: ControlModel = LuminousControlModel("luminous-supreme-control-20240215"),
    ):
        super().__init__()
        self._model = model
        self._chunk_task = ChunkTask(model, max_tokens_per_chunk)
        self._multi_chunk_qa = multi_chunk_qa or MultipleChunkQa(model=model)
        self._k = k
        self._language_detector = DetectLanguage(threshold=0.5)

    def do_run(
        self, input: LongContextQaInput, task_span: TaskSpan
    ) -> MultipleChunkQaOutput:
        chunk_output = self._chunk_task.run(ChunkInput(text=input.text), task_span)
        retriever = QdrantInMemoryRetriever(
            self._model._client,
            documents=[
                Document(
                    text=c,
                )
                for c in chunk_output.chunks
            ],
            k=self._k,
            threshold=0.5,
        )

        search_output = Search(retriever).run(
            SearchInput(query=input.question), task_span
        )

        multi_chunk_qa_input = MultipleChunkQaInput(
            chunks=[
                Chunk(result.document_chunk.text) for result in search_output.results
            ],
            question=input.question,
            language=input.language,
        )
        qa_output = self._multi_chunk_qa.run(multi_chunk_qa_input, task_span)
        return qa_output
