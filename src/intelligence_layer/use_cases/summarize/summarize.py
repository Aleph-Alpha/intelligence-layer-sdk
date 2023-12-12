from statistics import mean
from typing import Iterable, Sequence, Union

from pydantic import BaseModel

from intelligence_layer.core import EvaluationRepository, Evaluator
from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.evaluation.graders import BleuGrader, RougeGrader
from intelligence_layer.core.task import Task


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
    chunk: Chunk
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

    chunk: Chunk
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
        rouge: rougly corresponds to recall
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
        evaluation: The actual evaluations
    """

    aggregate_bleu: float
    aggregate_rouge: float
    evaluations: Sequence[SummarizeEvaluation]


class SingleChunkSummarizeEvaluator(
    Evaluator[
        SingleChunkSummarizeInput,
        SummarizeOutput,
        str,
        SummarizeEvaluation,
        AggregatedSummarizeEvaluation,
    ]
):
    def __init__(
        self,
        task: Task[SingleChunkSummarizeInput, SummarizeOutput],
        repository: EvaluationRepository,
    ) -> None:
        super().__init__(task, repository)
        self.bleu_grader = BleuGrader()
        self.rouge_grader = RougeGrader()

    # mypy expects *args where this method only uses one output
    def do_evaluate( # type: ignore
        self,
        input: SingleChunkSummarizeInput,
        expected_output: str,
        output: SummarizeOutput, 
    ) -> SummarizeEvaluation:
        bleu_score = self.bleu_grader.calculate_bleu(output.summary, expected_output)
        rouge_score = self.rouge_grader.calculate_rouge(output.summary, expected_output)

        return SummarizeEvaluation(
            bleu=bleu_score, rouge=rouge_score.recall, output=output
        )

    def aggregate(
        self, evaluations: Iterable[SummarizeEvaluation]
    ) -> AggregatedSummarizeEvaluation:
        evaluations_list = list(evaluations)
        if len(evaluations_list) != 0:
            bleu_avg = mean(eval.bleu for eval in evaluations_list)
            rouge_avg = mean(eval.rouge for eval in evaluations_list)
        else:
            bleu_avg = 0.0
            rouge_avg = 0.0
        return AggregatedSummarizeEvaluation(
            aggregate_bleu=bleu_avg,
            aggregate_rouge=rouge_avg,
            evaluations=evaluations_list,
        )


class LongContextSummarizeEvaluator(
    Evaluator[
        LongContextSummarizeInput,
        LongContextSummarizeOutput,
        str,
        SummarizeEvaluation,
        AggregatedSummarizeEvaluation,
    ]
):
    def __init__(
        self,
        task: Task[LongContextSummarizeInput, LongContextSummarizeOutput],
        repository: EvaluationRepository,
    ) -> None:
        super().__init__(task, repository)
        self.bleu_grader = BleuGrader()
        self.rouge_grader = RougeGrader()

    # mypy expects *args where this method only uses one output
    def do_evaluate( # type: ignore
        self,
        input: LongContextSummarizeInput,
        expected_output: str,
        output: LongContextSummarizeOutput,
    ) -> SummarizeEvaluation:
        joint_summary = " ".join(
            partial_summary.summary for partial_summary in output.partial_summaries
        )
        bleu_score = self.bleu_grader.calculate_bleu(joint_summary, expected_output)
        rouge_score = self.rouge_grader.calculate_rouge(joint_summary, expected_output)

        return SummarizeEvaluation(
            bleu=bleu_score, rouge=rouge_score.recall, output=output
        )

    def aggregate(
        self, evaluations: Iterable[SummarizeEvaluation]
    ) -> AggregatedSummarizeEvaluation:
        evaluations_list = list(evaluations)
        if len(evaluations_list) != 0:
            bleu_avg = mean(eval.bleu for eval in evaluations_list)
            rouge_avg = mean(eval.rouge for eval in evaluations_list)
        else:
            bleu_avg = 0.0
            rouge_avg = 0.0
        return AggregatedSummarizeEvaluation(
            aggregate_bleu=bleu_avg,
            aggregate_rouge=rouge_avg,
            evaluations=evaluations_list,
        )
