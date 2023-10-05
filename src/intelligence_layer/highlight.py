from math import exp
from typing import Optional, Sequence

from aleph_alpha_client import Client, ExplanationRequest, ExplanationResponse, TextScore, Prompt
import numpy as np
from pydantic import BaseModel

from intelligence_layer.task import DebugLog, LogLevel, Task


def sigmoid(x):
  return 1 / (1 + exp(-x))


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


class HighlightOutput(BaseModel):
    """Output of for a highlight task"""
    highlights: Sequence[str]
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
        aa_explanation = self._aa_explain(prompt=input.prompt, target=input.target, debug_log=debug_log)
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
    
    def _determine_highlight_range(self, prompt: str, highlight_text: str) -> TextRange:
        return TextRange.from_prompt_and_highlight_text(
            prompt=prompt, highlight_text=highlight_text
        )

    def _aa_explain(self, prompt: str, target: str, debug_log: DebugLog) -> ExplanationResponse:
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
        overlapping = [
            text_score
            for text_score in scores
            if isinstance(text_score, TextScore) and text_range.overlaps(text_score)
        ]
        debug_log.info(
            "Explanation-Scores",
            [
                {
                    "text": self._chop_highlight(text, text_score, text_range),
                    "score": text_score.score,
                }
                for text_score in overlapping
            ],
        )
        best_test_score = max(text_score.score for text_score in overlapping)
        return [
            self._chop_highlight(text, text_score, text_range)
            for text_score in overlapping
            if text_score.score == best_test_score
        ]

    def _chop_highlight(self, text: str, score: TextScore, range: TextRange) -> str:
        start = score.start - range.start
        return text[max(0, start) : start + score.length]
