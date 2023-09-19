import warnings
from collections import defaultdict
from typing import Iterable, Mapping, NewType, Sequence

from pydantic import BaseModel

from intelligence_layer.core import TextChunk
from intelligence_layer.evaluation import (
    AggregationLogic,
    Example,
    MeanAccumulator,
    SingleOutputEvaluationLogic,
)

Probability = NewType("Probability", float)


class ClassifyInput(BaseModel):
    """Input for a classification task.

    Attributes:
        chunk: text to be classified.
        labels: Possible labels the model will choose a label from
    """

    chunk: TextChunk
    labels: frozenset[str]


class SingleLabelClassifyOutput(BaseModel):
    """Output for a single label classification task.

    Attributes:
        scores: Mapping of the provided label (key) to corresponding score (value).
            The score represents how sure the model is that this is the correct label.
            This will be a value between 0 and 1.
            The sum of all probabilities will be 1.
    """

    scores: Mapping[str, Probability]

    @property
    def sorted_scores(self) -> list[tuple[str, Probability]]:
        return sorted(self.scores.items(), key=lambda item: item[1], reverse=True)


class MultiLabelClassifyOutput(BaseModel):
    """Output for a multi label classification task.

    Attributes:
        scores: Mapping of the provided label (key) to corresponding score (value).
            The score represents how sure the model is that this is the correct label.
            This will be a value between 0 and 1.
            There is not constraint on the sum of the individual probabilities.
    """

    scores: Mapping[str, Probability]


class SingleLabelClassifyEvaluation(BaseModel):
    """The evaluation of a single label classification run.

    Attributes:
        correct: Was the highest scoring class from the output in the set of "correct classes".
        predicted: The predicted label.
        expected: The expected label.
        expected_label_missing: Whether the expected label was missing from the possible set of
            labels in the task's input.
    """

    correct: bool
    predicted: str
    expected: str
    expected_label_missing: bool


class AggregatedLabelInfo(BaseModel):
    expected_count: int
    predicted_count: int


class AggregatedSingleLabelClassifyEvaluation(BaseModel):
    """The aggregated evaluation of a single label classify implementation against a dataset.

    Attributes:
        percentage_correct: Percentage of answers that were considered to be correct.
        confusion_matrix: A matrix showing the predicted classifications vs the expected classifications.
        by_label: Each label along side the counts how often it was expected or predicted.
        missing_labels: Each expected label which is missing in the set of possible labels in the task input and the number of its occurrences.
    """

    percentage_correct: float
    confusion_matrix: Mapping[tuple[str, str], int]
    by_label: Mapping[str, AggregatedLabelInfo]
    missing_labels: Mapping[str, int]


class SingleLabelClassifyAggregationLogic(
    AggregationLogic[
        SingleLabelClassifyEvaluation, AggregatedSingleLabelClassifyEvaluation
    ]
):
    def aggregate(
        self, evaluations: Iterable[SingleLabelClassifyEvaluation]
    ) -> AggregatedSingleLabelClassifyEvaluation:
        acc = MeanAccumulator()
        missing_labels: dict[str, int] = defaultdict(int)
        confusion_matrix: dict[tuple[str, str], int] = defaultdict(int)
        by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for evaluation in evaluations:
            acc.add(1.0 if evaluation.correct else 0.0)
            if evaluation.expected_label_missing:
                missing_labels[evaluation.expected] += 1
            else:
                confusion_matrix[(evaluation.predicted, evaluation.expected)] += 1
                by_label[evaluation.predicted]["predicted"] += 1
                by_label[evaluation.expected]["expected"] += 1

        if len(missing_labels) > 0:
            warn_message = "[WARNING] There were examples with expected labels missing in the evaluation inputs. For a detailed list, see the 'statistics.missing_labels' field of the returned `AggregationOverview`."
            warnings.warn(warn_message, RuntimeWarning)

        return AggregatedSingleLabelClassifyEvaluation(
            percentage_correct=acc.extract(),
            confusion_matrix=confusion_matrix,
            by_label={
                label: AggregatedLabelInfo(
                    expected_count=counts["expected"],
                    predicted_count=counts["predicted"],
                )
                for label, counts in by_label.items()
            },
            missing_labels=missing_labels,
        )


class SingleLabelClassifyEvaluationLogic(
    SingleOutputEvaluationLogic[
        ClassifyInput,
        SingleLabelClassifyOutput,
        str,
        SingleLabelClassifyEvaluation,
    ]
):
    def do_evaluate_single_output(
        self,
        example: Example[ClassifyInput, str],
        output: SingleLabelClassifyOutput,
    ) -> SingleLabelClassifyEvaluation:
        if example.expected_output not in example.input.labels:
            warn_message = f"[WARNING] Example with ID '{example.id}' has expected label '{example.expected_output}', which is not part of the example's input labels."
            warnings.warn(warn_message, RuntimeWarning)

        predicted = output.sorted_scores[0][0]
        if predicted == example.expected_output:
            correct = True
        else:
            correct = False
        return SingleLabelClassifyEvaluation(
            correct=correct,
            predicted=predicted,
            expected=example.expected_output,
            expected_label_missing=example.expected_output not in example.input.labels,
        )


class MultiLabelClassifyEvaluation(BaseModel):
    """The evaluation of a single multi-label classification example.

    Attributes:
        tp: The classes that were expected and correctly predicted (true positives).
        tn: The classes that were not expected and correctly not predicted (true negatives).
        fp: The classes that were not expected and falsely predicted (false positives).
        fn: The classes that were expected and falsely not predicted (false negatives).
    """

    tp: frozenset[str]
    tn: frozenset[str]
    fp: frozenset[str]
    fn: frozenset[str]


class MultiLabelClassifyMetrics(BaseModel):
    """The relevant metrics resulting from a confusion matrix in a classification run.

    Attributes:
        precision: Proportion of correctly predicted classes to all predicted classes.
        recall: Proportion of correctly predicted classes to all expected classes.
        f1: Aggregated performance, formally the harmonic mean of precision and recall.
    """

    precision: float
    recall: float
    f1: float


class AggregatedMultiLabelClassifyEvaluation(BaseModel):
    """The aggregated evaluation of a multi-label classify dataset.

    Attributes:
        class_metrics: Mapping of all labels to their aggregated metrics.
        micro_avg: Calculated by considering the tp, tn, fp and fn for each class, adding them up and dividing by them.
        macro_avg: The metrics' mean across all classes.

    """

    class_metrics: Mapping[str, MultiLabelClassifyMetrics]
    micro_avg: MultiLabelClassifyMetrics
    macro_avg: MultiLabelClassifyMetrics


class MultiLabelClassifyAggregationLogic(
    AggregationLogic[
        MultiLabelClassifyEvaluation, AggregatedMultiLabelClassifyEvaluation
    ]
):
    def aggregate(
        self, evaluations: Iterable[MultiLabelClassifyEvaluation]
    ) -> AggregatedMultiLabelClassifyEvaluation:
        label_confusion_matrix: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        )

        for evaluation in evaluations:
            for tp in evaluation.tp:
                label_confusion_matrix[tp]["tp"] += 1
            for tn in evaluation.tn:
                label_confusion_matrix[tn]["tn"] += 1
            for fp in evaluation.fp:
                label_confusion_matrix[fp]["fp"] += 1
            for fn in evaluation.fn:
                label_confusion_matrix[fn]["fn"] += 1

        class_metrics = {}
        sum_tp, sum_fp, sum_fn = 0, 0, 0
        sum_precision, sum_recall, sum_f1 = 0.0, 0.0, 0.0

        for label, confusion_matrix in label_confusion_matrix.items():
            precision = (
                confusion_matrix["tp"]
                / (confusion_matrix["tp"] + confusion_matrix["fp"])
                if confusion_matrix["tp"] + confusion_matrix["fp"]
                else 0
            )
            recall = (
                confusion_matrix["tp"]
                / (confusion_matrix["tp"] + confusion_matrix["fn"])
                if confusion_matrix["tp"] + confusion_matrix["fn"]
                else 0
            )
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if precision + recall
                else 0
            )

            class_metrics[label] = MultiLabelClassifyMetrics(
                precision=precision, recall=recall, f1=f1
            )

            sum_tp += confusion_matrix["tp"]
            sum_fp += confusion_matrix["fp"]
            sum_fn += confusion_matrix["fn"]
            sum_precision += precision
            sum_recall += recall
            sum_f1 += f1

        try:
            micro_avg = MultiLabelClassifyMetrics(
                precision=sum_tp / (sum_tp + sum_fp),
                recall=sum_tp / (sum_tp + sum_fn),
                f1=(2 * (sum_tp / (sum_tp + sum_fp)) * (sum_tp / (sum_tp + sum_fn)))
                / ((sum_tp / (sum_tp + sum_fp)) + (sum_tp / (sum_tp + sum_fn))),
            )
        except ZeroDivisionError:
            micro_avg = MultiLabelClassifyMetrics(
                precision=0,
                recall=0,
                f1=0,
            )
        macro_avg = MultiLabelClassifyMetrics(
            precision=sum_precision / len(class_metrics),
            recall=sum_recall / len(class_metrics),
            f1=sum_f1 / len(class_metrics),
        )

        return AggregatedMultiLabelClassifyEvaluation(
            class_metrics=class_metrics, micro_avg=micro_avg, macro_avg=macro_avg
        )


class MultiLabelClassifyEvaluationLogic(
    SingleOutputEvaluationLogic[
        ClassifyInput,
        MultiLabelClassifyOutput,
        Sequence[str],
        MultiLabelClassifyEvaluation,
    ]
):
    def __init__(
        self,
        threshold: float = 0.55,
    ):
        super().__init__()
        self.threshold = threshold

    def do_evaluate_single_output(
        self,
        example: Example[ClassifyInput, Sequence[str]],
        output: MultiLabelClassifyOutput,
    ) -> MultiLabelClassifyEvaluation:
        predicted_classes = frozenset(
            label for label, score in output.scores.items() if score > self.threshold
        )
        expected_classes = frozenset(example.expected_output)
        tp = predicted_classes & expected_classes
        tn = (example.input.labels - predicted_classes) & (
            example.input.labels - expected_classes
        )
        fp = (example.input.labels - expected_classes) - (
            example.input.labels - predicted_classes
        )
        fn = expected_classes - predicted_classes

        return MultiLabelClassifyEvaluation(tp=tp, tn=tn, fp=fp, fn=fn)
