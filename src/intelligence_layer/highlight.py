import math
from typing import Sequence, Union

from aleph_alpha_client import Client, ExplanationRequest, ExplanationResponse, TextScore, Prompt
from pydantic import BaseModel

from intelligence_layer.task import DebugLog, LogLevel, Task


class TextRange(BaseModel):
    start: int
    end: int

    def contains(self, pos: int) -> bool:
        return self.start <= pos < self.end

    def overlaps(self, score: TextScore) -> bool:
        return self.contains(score.start) or self.contains(score.start + score.length)
    
    def get_text(self, prompt: str) -> str:
        return prompt[self.start : self.end]


class HighlightInput(BaseModel):
    """Input for a highlight task"""
    prompt: str
    target: str
    highlight_range: TextRange


class ScoredHighlight(BaseModel):
    text: str
    score: Union[float, int]
    sigmoid_score: Union[float, int]
    text_range: TextRange


class HighlightOutput(BaseModel):
    """Output of for a highlight task"""
    highlights: Sequence[ScoredHighlight]
    debug_log: DebugLog
    """Provides key steps, decisions, and intermediate outputs of a task's process."""


class Highlight(Task[HighlightInput, HighlightOutput]):
    client: Client

    def __init__(self, client: Client, log_level: LogLevel) -> None:
        """Initializes the Task.

        Args:
        - client: the aleph alpha client
        """
        super().__init__()
        self.client = client
        self.log_level = log_level    
    
    def run(
        self, input: HighlightInput
    ) -> HighlightOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        aa_explanation = self._explain(prompt=input.prompt, target=input.target, debug_log=debug_log)
        highlights = self._to_highlights(
            explanation=aa_explanation,
            text=input.highlight_range.get_text(input.prompt),
            text_range=input.highlight_range,
            debug_log=debug_log
        )
        return HighlightOutput(
            highlights=highlights,
            debug_log=debug_log
        )

    def _explain(self, prompt: str, target: str, debug_log: DebugLog) -> ExplanationResponse:
        request = ExplanationRequest(Prompt.from_text(prompt), target)
        response = self.client.explain(request, "luminous-base-control")
        debug_log.debug(
            "Explanation Request/Response", {"request": request, "response": response}
        )
        return response

    def _to_highlights(
        self,
        explanation: ExplanationResponse,
        text: str,
        text_range: TextRange,
        debug_log: DebugLog,
    ) -> Sequence[str]:
        scores = explanation.explanations[0].items[0].scores
        overlapping_scores = [
            text_score
            for text_score in scores
            if isinstance(text_score, TextScore) and text_range.overlaps(text_score)
        ]
        debug_log.info(
            "Explanation-Scores",
            [
                {
                    "highlight": self._to_highlight(text, text_score, text_range),
                    "score": text_score.score,
                }
                for text_score in overlapping_scores
            ],
        )
        relevant_scores = self._filter_values(overlapping_scores, 1, debug_log)
        return [
            self._to_highlight(text, text_score, text_range)
            for text_score in relevant_scores
        ]

    @staticmethod
    def _z_scores(data: Sequence[float], debug_log: DebugLog) -> Sequence[float]:
        mean = sum(data) / len(data)
        std_dev = math.sqrt(sum([(x - mean) ** 2 for x in data]) / (len(data) - 1))
        z_scores = [(x - mean) / std_dev for x in data]
        debug_log.info(
            "Explanation-Statistics",
            {
                "mean": mean,
                "std_dev": std_dev,
                "z_scores": z_scores
            }
        )
        return z_scores

    def _filter_values(self, data: Sequence[TextScore], z_score_limit: Union[float, int], debug_log: DebugLog) -> Sequence[TextScore]:
        z_scores = self._z_scores([d.score for d in data], debug_log)
        return [d for d, z in zip(data, z_scores) if abs(z) >= z_score_limit]

    @staticmethod
    def _sigmoid(x: Union[float, int]) -> Union[float, int]:
        return 1 / (1 + math.exp(-x))

    def _to_highlight(self, text: str, score: TextScore, range: TextRange) -> str:
        start = score.start - range.start
        start_idx, end_idx = max(0, start), start + score.length
        return ScoredHighlight(
            text=text[start_idx:end_idx],
            score=score.score,
            sigmoid_score=self._sigmoid(score.score),
            text_range=TextRange(start=start_idx, end=end_idx)
        )
