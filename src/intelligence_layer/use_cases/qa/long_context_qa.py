from pydantic import BaseModel

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
)
from intelligence_layer.core import (
    Chunk,
    ChunkInput,
    ChunkOutput,
    ControlModel,
    Language,
    LuminousControlModel,
    Task,
    TaskSpan,
    TextChunk,
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
        multi_chunk_qa: task used to produce answers for each relevant chunk generated
            by the chunk-task for the given input. Defaults to :class:`MultipleChunkQa` .
        chunk: task used to chunk the input. Defaults to :class:`Chunk` .
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
        chunk: Task[ChunkInput, ChunkOutput] | None = None,
        k: int = 4,
        model: ControlModel | None = None,
    ):
        super().__init__()
        self._model = model or LuminousControlModel("luminous-supreme-control")
        self._chunk_task = chunk or Chunk(self._model, 1024)
        self._multi_chunk_qa = multi_chunk_qa or MultipleChunkQa(
            merge_answers_model=self._model
        )
        self._k = k

    def do_run(
        self, input: LongContextQaInput, task_span: TaskSpan
    ) -> MultipleChunkQaOutput:
        chunk_output = self._chunk_task.run(ChunkInput(text=input.text), task_span)
        retriever = QdrantInMemoryRetriever(
            client=self._model._client,
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
                TextChunk(result.document_chunk.text)
                for result in search_output.results
            ],
            question=input.question,
            language=input.language,
        )
        qa_output = self._multi_chunk_qa.run(multi_chunk_qa_input, task_span)
        return qa_output
