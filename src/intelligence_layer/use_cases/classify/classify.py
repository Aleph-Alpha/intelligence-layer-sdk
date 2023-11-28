from typing import Iterable, Mapping, NewType, Sequence, Union

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
    """Output for a single label classification task.

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
    """The evaluation of a multi-label classification run.

    Attributes:
        correct: TODO.
    """

    tp: frozenset[str]
    tn: frozenset[str]
    fp: frozenset[str]
    fn: frozenset[str]


class MultiLabelClassifyClassMetrics(BaseModel):
    """TODO"""

    precision: float
    recall: float
    f1: float


class AggregatedMultiLabelClassifyEvaluation(BaseModel):
    """TODO"""

    class_metrics: Mapping[str, MultiLabelClassifyClassMetrics]
    micro_avg: float
    macro_avg: float
    weighted_avg: float
    samples_avg: float


class MultiLabelClassifyEvaluator(
    Evaluator[
        ClassifyInput,
        MultiLabelClassifyOutput,
        Sequence[str],
        MultiLabelClassifyEvaluation,
        AggregatedSingleLabelClassifyEvaluation,
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
    ) -> SingleLabelClassifyEvaluation:
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
