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
from intelligence_layer.available_models import ControlModels

from intelligence_layer.task import DebugLog, LogLevel, Task


class SingleChunkQaInput(BaseModel):
    chunk: str
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


class SingleChunkQa(Task[SingleChunkQaInput, QaOutput]):
    """
    Perform Question answering on a text chunk that fits into the contex length (<2048 tokens for text prompt, question and answer)
    """

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

    def run(self, input: SingleChunkQaInput) -> QaOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        prompt_text, text_range = self._prompt_text(
            input.chunk, input.question, NO_ANSWER_TEXT
        )
        output = self._complete(prompt_text, debug_log)
        explanation = self._explain(prompt_text, output.completion(), debug_log)
        return QaOutput(
            answer=self._no_answer_to_none(output.completion().strip()),
            highlights=self._to_highlights(
                explanation, input.chunk, text_range, debug_log
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


if __name__ == "__main__":
    import os
    token = os.getenv("AA_TOKEN")
    client = Client(token=token)

    qa = SingleChunkQa(client, "info")
    
    input = SingleChunkQaInput(
        chunk="Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916.",
        question="What is the name of Paul Nicolas' brother?",
    )
    output = qa.run(input)

    assert output.answer
    assert "Henri" in output.answer
    assert any("Henri" in highlight for highlight in output.highlights)
    assert len(output.highlights) == 1