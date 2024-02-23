import statistics
from typing import Sequence

from pydantic import BaseModel
from qdrant_client.http.models import models

from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    QdrantInMemoryRetriever,
    RetrieverType,
)
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan
from intelligence_layer.use_cases.classify.classify import (
    ClassifyInput,
    MultiLabelClassifyOutput,
    Probability,
)
from intelligence_layer.use_cases.search.search import SearchOutput


class QdrantSearchInput(BaseModel):
    """The input for a `QdrantSearch` task.

    Attributes:
        query: The text to be searched with.
        filter: Conditions to filter by as offered by Qdrant.
    """

    query: str
    filter: models.Filter


class QdrantSearch(Task[QdrantSearchInput, SearchOutput[int]]):
    """Performs search to find documents using QDrant filtering methods.

    Given a query, this task will utilize a retriever to fetch relevant text search results.
    Contrary to `Search`, this `Task` offers the option to filter.

    Args:
        in_memory_retriever: Implements logic to retrieve matching texts to the query.

    Example:
        >>> import os
        >>> from intelligence_layer.connectors import (
        ...     LimitedConcurrencyClient,
        ... )
        >>> from intelligence_layer.connectors import Document
        >>> from intelligence_layer.connectors import (
        ...     QdrantInMemoryRetriever,
        ... )
        >>> from intelligence_layer.core import InMemoryTracer
        >>> from intelligence_layer.use_cases import (
        ...     QdrantSearch,
        ...     QdrantSearchInput,
        ... )
        >>> from qdrant_client.http.models import models

        >>> client = LimitedConcurrencyClient.from_env()
        >>> documents = [
        ...     Document(
        ...         text="West and East Germany reunited in 1990.", metadata={"title": "Germany"}
        ...     )
        ... ]
        >>> retriever = QdrantInMemoryRetriever(client, documents, 3)
        >>> task = QdrantSearch(retriever)
        >>> input = QdrantSearchInput(
        ...     query="When did East and West Germany reunite?",
        ...     filter=models.Filter(
        ...         must=[
        ...             models.FieldCondition(
        ...                 key="metadata.title",
        ...                 match=models.MatchValue(value="Germany"),
        ...             ),
        ...         ]
        ...     ),
        ... )
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    def __init__(self, in_memory_retriever: QdrantInMemoryRetriever):
        super().__init__()
        self._in_memory_retriever = in_memory_retriever

    def do_run(
        self, input: QdrantSearchInput, task_span: TaskSpan
    ) -> SearchOutput[int]:
        results = self._in_memory_retriever.get_filtered_documents_with_scores(
            input.query, input.filter
        )
        return SearchOutput(results=results)


class LabelWithExamples(BaseModel):
    """Defines a label and the list of examples making it up.

    Attributes:
        name: Name of the label.
        examples: The texts defining the example. Should be similar in structure
            and semantics to the texts to be classified on inference.
    """

    name: str
    examples: Sequence[str]


class EmbeddingBasedClassify(Task[ClassifyInput, MultiLabelClassifyOutput]):
    """Task that classifies a given input text based on examples.

    The input contains a complete set of all possible labels. The output will return a score
    for each possible label. Scores will be between 0 and 1 but do not have to add up to one.
    On initiation, provide a list of examples for each label.

    This methodology works best with a larger number of examples per label and with labels
    that consist of easily definable semantic clusters.

    Args:
        labels_with_examples: Examples to be used for classification.
        client: Aleph Alpha client instance for running model related API calls.
        scoring: Configure how to calculate the final score.

    Attributes:
        METADATA_LABEL_NAME: The metadata field name for 'label' in the retriever.

    Example:
        >>> from intelligence_layer.connectors.limited_concurrency_client import (
        ...     LimitedConcurrencyClient,
        ... )
        >>> from intelligence_layer.core import Chunk
        >>> from intelligence_layer.core.tracer import InMemoryTracer
        >>> from intelligence_layer.use_cases.classify.classify import ClassifyInput
        >>> from intelligence_layer.use_cases.classify.embedding_based_classify import (
        ...     EmbeddingBasedClassify,
        ...     LabelWithExamples,
        ... )


        >>> labels_with_examples = [
        ...     LabelWithExamples(
        ...         name="positive",
        ...         examples=[
        ...             "I really like this.",
        ...         ],
        ...     ),
        ...     LabelWithExamples(
        ...         name="negative",
        ...         examples=[
        ...             "I really dislike this.",
        ...         ],
        ...     ),
        ... ]
        >>> client = LimitedConcurrencyClient.from_env()
        >>> task = EmbeddingBasedClassify(client, labels_with_examples)
        >>> input = ClassifyInput(chunk=Chunk("This is a happy text."), labels=frozenset({"positive", "negative"}))
        >>> tracer = InMemoryTracer()
        >>> output = task.run(input, tracer)
    """

    METADATA_LABEL_NAME = "label"

    def __init__(
        self,
        client: AlephAlphaClientProtocol,
        labels_with_examples: Sequence[LabelWithExamples],
        top_k_per_label: int = 5,
    ) -> None:
        super().__init__()
        self._labels_with_examples = labels_with_examples
        documents = self._labels_with_examples_to_documents(labels_with_examples)
        self._scoring = top_k_per_label
        retriever = QdrantInMemoryRetriever(
            client,
            documents=documents,
            k=top_k_per_label,
            retriever_type=RetrieverType.SYMMETRIC,
        )
        self._qdrant_search = QdrantSearch(retriever)

    def do_run(
        self, input: ClassifyInput, task_span: TaskSpan
    ) -> MultiLabelClassifyOutput:
        self._validate_input_labels(input)
        results_per_label = [
            self._label_search(input.chunk, label, task_span) for label in input.labels
        ]
        scores = self._calculate_scores(results_per_label)
        return MultiLabelClassifyOutput(
            scores={lang: Probability(s) for lang, s in zip(input.labels, scores)}
        )

    def _labels_with_examples_to_documents(
        self, classes_with_examples: Sequence[LabelWithExamples]
    ) -> Sequence[Document]:
        return [
            Document(
                text=e,
                metadata={self.METADATA_LABEL_NAME: class_with_examples.name},
            )
            for class_with_examples in classes_with_examples
            for e in class_with_examples.examples
        ]

    def _validate_input_labels(self, input: ClassifyInput) -> None:
        available_labels = set(
            class_with_examples.name
            for class_with_examples in self._labels_with_examples
        )
        unknown_labels = input.labels - available_labels
        if unknown_labels:
            raise ValueError(f"Got unexpected labels: {', '.join(unknown_labels)}.")

    def _label_search(
        self, chunk: Chunk, label: str, task_span: TaskSpan
    ) -> SearchOutput[int]:
        search_input = QdrantSearchInput(
            query=chunk,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=f"metadata.{self.METADATA_LABEL_NAME}",
                        match=models.MatchValue(value=label),
                    ),
                ]
            ),
        )
        return self._qdrant_search.run(search_input, task_span)

    def _calculate_scores(
        self, results_per_label: Sequence[SearchOutput[int]]
    ) -> Sequence[float]:
        return [
            (statistics.mean(r.score for r in r_per_l.results) + 1) / 2
            for r_per_l in results_per_label
        ]
