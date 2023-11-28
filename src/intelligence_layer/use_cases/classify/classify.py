from collections import defaultdict
from typing import Iterable, Mapping, NewType, Sequence

from pydantic import BaseModel

from intelligence_layer.core import EvaluationRepository, Evaluator
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.task import Task

Probability = NewType("Probability", float)


class ClassifyInput(BaseModel):
    """Input for a classification task.

    Attributes:
        chunk: text to be classified.
        labels: Possible labels the model will choose a label from
    """

    chunk: Chunk
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
    """

    correct: bool


class AggregatedSingleLabelClassifyEvaluation(BaseModel):
    """The aggregated evaluation of a single label classify implementation against a dataset.

    Attributes:
        percentage_correct: Percentage of answers that were considered to be correct
    """

    percentage_correct: float


class SingleLabelClassifyEvaluator(
    Evaluator[
        ClassifyInput,
        SingleLabelClassifyOutput,
        Sequence[str],
        SingleLabelClassifyEvaluation,
        AggregatedSingleLabelClassifyEvaluation,
    ]
):
    def __init__(
        self,
        task: Task[ClassifyInput, SingleLabelClassifyOutput],
        repository: EvaluationRepository,
    ):
        super().__init__(task, repository)

    def do_evaluate(
        self,
        input: ClassifyInput,
        output: SingleLabelClassifyOutput,
        expected_output: Sequence[str],
    ) -> SingleLabelClassifyEvaluation:
        sorted_classes = sorted(
            output.scores.items(), key=lambda item: item[1], reverse=True
        )
        if sorted_classes[0][0] in expected_output:
            correct = True
        else:
            correct = False
        return SingleLabelClassifyEvaluation(correct=correct)

    def aggregate(
        self, evaluations: Iterable[SingleLabelClassifyEvaluation]
    ) -> AggregatedSingleLabelClassifyEvaluation:
        evaluations_list = list(evaluations)
        if len(evaluations_list) != 0:
            correct_answers = len(
                [eval.correct for eval in evaluations_list if eval.correct is True]
            ) / len(evaluations_list)
        else:
            correct_answers = 0
        return AggregatedSingleLabelClassifyEvaluation(
            percentage_correct=correct_answers
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


class MultiLabelClassifyEvaluator(
    Evaluator[
        ClassifyInput,
        MultiLabelClassifyOutput,
        Sequence[str],
        MultiLabelClassifyEvaluation,
        AggregatedMultiLabelClassifyEvaluation,
    ]
):
    def __init__(
        self,
        task: Task[ClassifyInput, MultiLabelClassifyOutput],
        repository: EvaluationRepository,
    ):
        super().__init__(task, repository)

    def do_evaluate(
        self,
        input: ClassifyInput,
        output: MultiLabelClassifyOutput,
        expected_output: Sequence[str],
    ) -> MultiLabelClassifyEvaluation:
        predicted_classes = frozenset(
            label for label, score in output.scores.items() if score > 0.5
        )
        expected_classes = frozenset(expected_output)
        tp = predicted_classes & expected_classes
        tn = (input.labels - predicted_classes) & (input.labels - expected_classes)
        fp = (input.labels - expected_classes) - (input.labels - predicted_classes)
        fn = expected_classes - predicted_classes

        return MultiLabelClassifyEvaluation(tp=tp, tn=tn, fp=fp, fn=fn)

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
            precision = confusion_matrix["tp"] / (
                confusion_matrix["tp"] + confusion_matrix["fp"]
            )
            recall = confusion_matrix["tp"] / (
                confusion_matrix["tp"] + confusion_matrix["fn"]
            )
            f1 = (2 * precision * recall) / (precision + recall)

            class_metrics[label] = MultiLabelClassifyMetrics(
                precision=precision, recall=recall, f1=f1
            )

            sum_tp += confusion_matrix["tp"]
            sum_fp += confusion_matrix["fp"]
            sum_fn += confusion_matrix["fn"]
            sum_precision += precision
            sum_recall += recall
            sum_f1 += f1

        micro_avg = MultiLabelClassifyMetrics(
            precision=sum_tp / (sum_tp + sum_fp),
            recall=sum_tp / (sum_tp + sum_fn),
            f1=(2 * (sum_tp / (sum_tp + sum_fp)) * (sum_tp / (sum_tp + sum_fn)))
            / ((sum_tp / (sum_tp + sum_fp)) + (sum_tp / (sum_tp + sum_fn))),
        )
        macro_avg = MultiLabelClassifyMetrics(
            precision=sum_precision / len(class_metrics),
            recall=sum_recall / len(class_metrics),
            f1=sum_f1 / len(class_metrics),
        )

        return AggregatedMultiLabelClassifyEvaluation(
            class_metrics=class_metrics, micro_avg=micro_avg, macro_avg=macro_avg
        )
