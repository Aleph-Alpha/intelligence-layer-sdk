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


class ClassifyEvaluation(BaseModel):
    """The evaluation of a single label classification run.

    Attributes:
        correct: Was the highest scoring class from the output in the set of "correct classes".
        output: The actual output from the task run.
    """

    correct: bool
    output: Union[SingleLabelClassifyOutput, MultiLabelClassifyOutput]


class AggregatedClassifyEvaluation(BaseModel):
    """The aggregated evaluation of a single label classify implementation against a dataset.

    Attributes:
        percentage_correct: Percentage of answers that were considered to be correct
        evaluation: The actual evaluations
    """

    percentage_correct: float
    evaluations: Sequence[ClassifyEvaluation]


class SingleLabelClassifyEvaluator(
    Evaluator[
        ClassifyInput,
        SingleLabelClassifyOutput,
        Sequence[str],
        ClassifyEvaluation,
        AggregatedClassifyEvaluation,
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
    ) -> ClassifyEvaluation:
        sorted_classes = sorted(
            output.scores.items(), key=lambda item: item[1], reverse=True
        )
        if sorted_classes[0][0] in expected_output:
            correct = True
        else:
            correct = False
        return ClassifyEvaluation(correct=correct, output=output)

    def aggregate(
        self, evaluations: Iterable[ClassifyEvaluation]
    ) -> AggregatedClassifyEvaluation:
        evaluations_list = list(evaluations)
        if len(evaluations_list) != 0:
            correct_answers = len(
                [eval.correct for eval in evaluations_list if eval.correct is True]
            ) / len(evaluations_list)
        else:
            correct_answers = 0
        return AggregatedClassifyEvaluation(
            percentage_correct=correct_answers, evaluations=evaluations_list
        )


class MultiLabelClassifyEvaluator(
    Evaluator[
        ClassifyInput,
        MultiLabelClassifyOutput,
        Sequence[str],
        ClassifyEvaluation,
        AggregatedClassifyEvaluation,
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
    ) -> ClassifyEvaluation:
        sorted_classes = sorted(
            output.scores.items(), key=lambda item: item[1], reverse=True
        )
        if sorted_classes[0][0] in expected_output:
            correct = True
        else:
            correct = False
        return ClassifyEvaluation(correct=correct, output=output)

    def aggregate(
        self, evaluations: Iterable[ClassifyEvaluation]
    ) -> AggregatedClassifyEvaluation:
        evaluations_list = list(evaluations)
        if len(evaluations_list) != 0:
            correct_answers = len(
                [eval.correct for eval in evaluations_list if eval.correct is True]
            ) / len(evaluations_list)
        else:
            correct_answers = 0
        return AggregatedClassifyEvaluation(
            percentage_correct=correct_answers, evaluations=evaluations_list
        )
