import math
from typing import Sequence

from aleph_alpha_client import (
    Client,
    ExplanationRequest,
    ExplanationResponse,
    TextScore,
    Prompt,
)
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
    text_range: TextRange
    score: float
    sigmoid_score: float
    z_score: float


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

    def run(self, input: HighlightInput) -> HighlightOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        aa_explanation = self._explain(
            prompt=input.prompt, target=input.target, debug_log=debug_log
        )
        highlights = self._to_highlights(
            explanation=aa_explanation,
            prompt=input.prompt,
            highlight_range=input.highlight_range,
            debug_log=debug_log,
        )
        return HighlightOutput(highlights=highlights, debug_log=debug_log)

    def _explain(
        self, prompt: str, target: str, debug_log: DebugLog
    ) -> ExplanationResponse:
        request = ExplanationRequest(Prompt.from_text(prompt), target)
        response = self.client.explain(request, "luminous-base-control")
        debug_log.debug(
            "Explanation Request/Response", {"request": request, "response": response}
        )
        return response

    def _to_highlights(
        self,
        explanation: ExplanationResponse,
        prompt: str,
        highlight_range: TextRange,
        debug_log: DebugLog,
    ) -> Sequence[ScoredHighlight]:
        scores = [
            s
            for s in explanation.explanations[0].items[0].scores
            if isinstance(s, TextScore)
        ]
        debug_log.info(
            "All scored highlights",
            self._score_highlights(prompt, scores, debug_log),
        )
        overlapping = [
            text_score for text_score in scores if highlight_range.overlaps(text_score)
        ]
        highlights = self._score_highlights(prompt, overlapping, debug_log)
        return self._filter_highlights(highlights)

    def _score_highlights(
        self, prompt: str, scores: Sequence[TextScore], debug_log: DebugLog
    ) -> Sequence[ScoredHighlight]:
        z_scores = self._z_scores([s.score for s in scores], debug_log)
        return [
            self._to_highlight(prompt, score, z_score)
            for score, z_score in zip(scores, z_scores)
        ]

    @staticmethod
    def _z_scores(data: Sequence[float], debug_log: DebugLog) -> Sequence[float]:
        mean = sum(data) / len(data)
        std_dev = math.sqrt(sum([(x - mean) ** 2 for x in data]) / (len(data) - 1))
        debug_log.info("Highlight statistics", {"mean": mean, "std_dev": std_dev})
        return [(x - mean) / std_dev for x in data]

    def _to_highlight(
        self, prompt: str, score: TextScore, z_score: float
    ) -> ScoredHighlight:
        text_range = TextRange(start=score.start, end=score.start + score.length)
        return ScoredHighlight(
            text=text_range.get_text(prompt),
            text_range=text_range,
            score=score.score,
            sigmoid_score=self._sigmoid(score.score),
            z_score=z_score,
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _filter_highlights(
        self, scored_highlights: Sequence[ScoredHighlight], z_score_limit: float = 0.5
    ) -> Sequence[ScoredHighlight]:
        return [h for h in scored_highlights if abs(h.z_score) >= z_score_limit]
