from typing import Optional, Sequence, Tuple
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    ExplanationRequest,
    ExplanationResponse,
    Prompt,
    TextScore,
)
from pydantic import BaseModel
from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from intelligence_layer.avalible_models import ControlModels

from intelligence_layer.task import DebugLog, LogLevel, Task


class SingleDocumentQaInput(BaseModel):
    text: str
    question: str


class QaOutput(BaseModel):
    answer: Optional[str]
    highlights: Sequence[str]
    debug_log: DebugLog


class TextRange(BaseModel):
    start: int
    end: int

    def contains(self, pos: int) -> bool:
        return self.start <= pos < self.end

    def overlaps(self, score: TextScore) -> bool:
        return self.contains(score.start) or self.contains(score.start + score.length)


NO_ANSWER_TEXT = "NO_ANSWER_IN_TEXT"


class SingleDocumentQa(Task[SingleDocumentQaInput, QaOutput]):
    PREFIX_TEMPLATE_STR = """### Instruction:
{question} If there's no answer, say "{no_answer_text}".

### Input:
"""
    POSTFIX_TEMPLATE_STR = """

### Response:"""

    def __init__(
        self,
        client: Client,
        log_level: LogLevel,
        model: ControlModels = ControlModels.SUPREME_CONTROL,
    ):
        self.client = client
        self.log_level = log_level
        self.completion = Completion(client, log_level)
        self.model = model

    def run(self, input: SingleDocumentQaInput) -> QaOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        prompt_text, text_range = self._prompt_text(
            input.text, input.question, NO_ANSWER_TEXT
        )
        output = self._complete(prompt_text, debug_log)
        explanation = self._explain(prompt_text, output.completion(), debug_log)
        return QaOutput(
            answer=self._no_answer_to_none(output.completion().strip()),
            highlights=self._to_highlights(
                explanation, input.text, text_range, debug_log
            ),
            debug_log=debug_log,
        )

    def _no_answer_to_none(self, completion: str) -> Optional[str]:
        return completion if completion != NO_ANSWER_TEXT else None

    def _prompt_text(
        self, text: str, question: str, no_answer_text: str
    ) -> Tuple[str, TextRange]:
        prefix = self.PREFIX_TEMPLATE_STR.format(
            question=question, no_answer_text=no_answer_text
        )
        return prefix + text + self.POSTFIX_TEMPLATE_STR, TextRange(
            start=len(prefix), end=len(prefix) + len(text)
        )

    def _complete(self, prompt: str, debug_log: DebugLog) -> CompletionOutput:
        request = CompletionRequest(Prompt.from_text(prompt))
        output = self.completion.run(CompletionInput(request=request, model=self.model))
        debug_log.debug("Completion", output.debug_log)
        debug_log.info(
            "Completion Input/Output",
            {"prompt": prompt, "completion": output.completion()},
        )
        return output

    def _explain(
        self, prompt: str, target: str, debug_log: DebugLog
    ) -> ExplanationResponse:
        request = ExplanationRequest(Prompt.from_text(prompt), target)
        response = self.client.explain(request, self.model)
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
                    "text": self.chop_highlight(text, text_score, text_range),
                    "score": text_score.score,
                }
                for text_score in overlapping
            ],
        )
        best_test_score = max(text_score.score for text_score in overlapping)
        return [
            self.chop_highlight(text, text_score, text_range)
            for text_score in overlapping
            if text_score.score == best_test_score
        ]

    def chop_highlight(self, text: str, score: TextScore, range: TextRange) -> str:
        start = score.start - range.start
        return text[max(0, start) : start + score.length]
