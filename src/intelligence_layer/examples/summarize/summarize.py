from collections.abc import Iterable, Sequence
from typing import Union

from pydantic import BaseModel

from intelligence_layer.core import Language, TextChunk
from intelligence_layer.evaluation import (
    AggregationLogic,
    BleuGrader,
    Example,
    MeanAccumulator,
    RougeGrader,
    SingleOutputEvaluationLogic,
)


class LongContextSummarizeInput(BaseModel):
    """The input for a summarize-task for a text of any length.

    Attributes:
        text: A text of any length.
        language: The desired language of the summary. ISO 619 str with language e.g. en, fr, etc.
    """

    text: str
    language: Language = Language("en")


class PartialSummary(BaseModel):
    """The summary of a single chunk.

    Attributes:
        summary: The summary generated by the task.
        chunk: The source chunk.
        generated_tokens: The number of tokens generated for the summary
    """

    summary: str
    chunk: TextChunk
    generated_tokens: int


class LongContextSummarizeOutput(BaseModel):
    """The output of a summarize-task for a text of any length.

    Attributes:
        partial_summaries: Chunk-wise summaries.
    """

    partial_summaries: Sequence[PartialSummary]


class SingleChunkSummarizeInput(BaseModel):
    """The input for a summarize-task that only deals with a single chunk.

    Attributes:
        chunk: The text chunk to be summarized.
        language: The desired language of the summary. ISO 619 str with language e.g. en, fr, etc.
    """

    chunk: TextChunk
    language: Language = Language("en")


class SummarizeOutput(BaseModel):
    """The output of a summarize-task.

    Attributes:
        summary: The summary generated by the task.
        generated_tokens: The number of tokens generated for the summary.
    """

    summary: str
    generated_tokens: int


class SummarizeEvaluation(BaseModel):
    """The evaluation of a summarization run.

    Attributes:
        bleu: roughly corresponds to precision
        rouge: roughly corresponds to recall
        output: The actual output from the task run
    """

    bleu: float
    rouge: float
    output: Union[SummarizeOutput, LongContextSummarizeOutput]


class AggregatedSummarizeEvaluation(BaseModel):
    """The aggregated evaluation of a summarization implementation against a dataset.

    Attributes:
        aggregate_bleu: average over BLEU-scores
        aggregate_rouge: average over ROUGE-scores
    """

    aggregate_bleu: float
    aggregate_rouge: float


class SingleChunkSummarizeAggregationLogic(
    AggregationLogic[SummarizeEvaluation, AggregatedSummarizeEvaluation]
):
    def aggregate(
        self, evaluations: Iterable[SummarizeEvaluation]
    ) -> AggregatedSummarizeEvaluation:
        return aggregate_summarize_evaluation(evaluations)


class SingleChunkSummarizeEvaluationLogic(
    SingleOutputEvaluationLogic[
        SingleChunkSummarizeInput,
        SummarizeOutput,
        str,
        SummarizeEvaluation,
    ]
):
    def __init__(self) -> None:
        super().__init__()
        self.bleu_grader = BleuGrader()
        self.rouge_grader = RougeGrader()

    def do_evaluate_single_output(
        self,
        example: Example[SingleChunkSummarizeInput, str],
        output: SummarizeOutput,
    ) -> SummarizeEvaluation:
        bleu_score = self.bleu_grader.calculate_bleu(
            output.summary, example.expected_output
        )
        rouge_score = self.rouge_grader.calculate_rouge(
            output.summary, example.expected_output
        )

        return SummarizeEvaluation(
            bleu=bleu_score, rouge=rouge_score.recall, output=output
        )


class LongContextSummarizeAggregationLogic(
    AggregationLogic[SummarizeEvaluation, AggregatedSummarizeEvaluation]
):
    def aggregate(
        self, evaluations: Iterable[SummarizeEvaluation]
    ) -> AggregatedSummarizeEvaluation:
        return aggregate_summarize_evaluation(evaluations)


class LongContextSummarizeEvaluationLogic(
    SingleOutputEvaluationLogic[
        LongContextSummarizeInput,
        LongContextSummarizeOutput,
        str,
        SummarizeEvaluation,
    ]
):
    def __init__(self) -> None:
        super().__init__()
        self.bleu_grader = BleuGrader()
        self.rouge_grader = RougeGrader()

    def do_evaluate_single_output(
        self,
        example: Example[LongContextSummarizeInput, str],
        output: LongContextSummarizeOutput,
    ) -> SummarizeEvaluation:
        joint_summary = " ".join(
            partial_summary.summary for partial_summary in output.partial_summaries
        )
        bleu_score = self.bleu_grader.calculate_bleu(
            joint_summary, example.expected_output
        )
        rouge_score = self.rouge_grader.calculate_rouge(
            joint_summary, example.expected_output
        )

        return SummarizeEvaluation(
            bleu=bleu_score, rouge=rouge_score.recall, output=output
        )


def aggregate_summarize_evaluation(
    evaluations: Iterable[SummarizeEvaluation],
) -> AggregatedSummarizeEvaluation:
    acc_bleu = MeanAccumulator()
    acc_rouge = MeanAccumulator()
    for evaluation in evaluations:
        acc_bleu.add(evaluation.bleu)
        acc_rouge.add(evaluation.rouge)
    return AggregatedSummarizeEvaluation(
        aggregate_bleu=acc_bleu.extract(),
        aggregate_rouge=acc_rouge.extract(),
    )
