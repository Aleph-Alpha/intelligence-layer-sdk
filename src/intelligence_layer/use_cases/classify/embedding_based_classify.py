from enum import Enum
import statistics
from typing import Sequence
from aleph_alpha_client import Client

from pydantic import BaseModel
from qdrant_client.http.models import models

from intelligence_layer.connectors.retrievers.base_retriever import Document
from intelligence_layer.connectors.retrievers.in_memory_retriever import InMemoryRetriever, RetrieverType
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.task import Chunk, Probability, Task
from intelligence_layer.use_cases.classify.classify import ClassifyInput, ClassifyOutput
from intelligence_layer.use_cases.search.filter_search import FilterSearch, FilterSearchInput
from intelligence_layer.use_cases.search.search import SearchOutput


class EmbeddingBasedClassifyScoring(Enum):
    MAX = 1
    MEAN_TOP_5 = 5


class LabelWithExamples(BaseModel):
    name: str
    examples: Sequence[str]


class EmbeddingBasedClassify(Task[ClassifyInput, ClassifyOutput]):
    """Task that classifies a given input text based on examples provided on initiation."""

    METADATA_LABEL_NAME = "label"

    def __init__(self, labels_with_examples: Sequence[LabelWithExamples], client: Client, scoring: EmbeddingBasedClassifyScoring = EmbeddingBasedClassifyScoring.MEAN_TOP_5) -> None:
        super().__init__()
        self._labels_with_examples = labels_with_examples
        documents = self._labels_with_examples_to_documents(labels_with_examples)
        self._scoring = scoring
        retriever = InMemoryRetriever(
            client,
            documents=documents,
            k=len(documents),
            retriever_type=RetrieverType.SYMMETRIC
        )
        self._filter_search = FilterSearch(retriever)

    def run(self, input: ClassifyInput, logger: DebugLogger) -> ClassifyOutput:
        available_labels = set(class_with_examples.name for class_with_examples in self._labels_with_examples)
        unknown_labels = input.labels - available_labels
        if unknown_labels:
            raise ValueError(f"Got unexpected labels: {unknown_labels}")
        labels = list(input.labels) # converting to list to preserve order
        results_per_label = [self._label_search(input.chunk, label, logger) for label in labels]
        scores = self._calculate_scores(results_per_label)
        return ClassifyOutput(
            scores={
                l: Probability(s) for l, s in zip(labels, scores)
            }
        )
        
    def _labels_with_examples_to_documents(self, classes_with_examples: Sequence[LabelWithExamples]) -> Sequence[Document]:
        return [
            Document(
                text=e,
                metadata={self.METADATA_LABEL_NAME: class_with_examples.name}
            ) for class_with_examples in classes_with_examples for e in class_with_examples.examples
        ]

    def _label_search(self, chunk: Chunk, label: str, logger: DebugLogger) -> SearchOutput:
        search_input = FilterSearchInput(
            query=chunk,
            limit=self._scoring.value,
            filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{self.METADATA_LABEL_NAME}",
                    match=models.MatchValue(value=label),
                ),
            ]
        )
        )
        return self._filter_search.run(search_input, logger)
    
    def _calculate_scores(self, results_per_label: Sequence[SearchOutput]) -> Sequence[float]:
        return [
            statistics.mean(r.score for r in r_per_l.results) for r_per_l in results_per_label
        ]
