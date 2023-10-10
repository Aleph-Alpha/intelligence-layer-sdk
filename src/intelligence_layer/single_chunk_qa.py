from typing import Optional, Sequence, Tuple
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    ExplanationRequest,
    ExplanationResponse,
    Prompt,
    Text,
    TextPromptItemExplanation,
    TextScore,
)
from aleph_alpha_client.explanation import TextScoreWithRaw
from pydantic import BaseModel

from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from intelligence_layer.prompt_template import (
    PromptRange,
    PromptTemplate,
    PromptWithMetadata,
    TextCursor,
)
from intelligence_layer.task import DebugLog, LogLevel, Task


class SingleChunkQaInput(BaseModel):
    chunk: str
    question: str


class SingleChunkQaOutput(BaseModel):
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


class SingleChunkQa(Task[SingleChunkQaInput, SingleChunkQaOutput]):
    PROMPT_TEMPLATE_STR = """### Instruction:
{{question}}
If there's no answer, say "{{no_answer_text}}".

### Input:
{% promptrange text %}{{text}}{% endpromptrange %}

### Response:"""
    NO_ANSWER_STR = "NO_ANSWER_IN_TEXT"

    def __init__(
        self,
        client: Client,
        log_level: LogLevel,
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.log_level = log_level
        self.completion = Completion(client, log_level)
        self.model = model

    def run(self, input: SingleChunkQaInput) -> SingleChunkQaOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        prompt_with_metadata = self._to_prompt_with_metadata(
            input.chunk, input.question
        )
        output = self._complete(prompt_with_metadata.prompt, debug_log)
        explanation = self._explain(
            prompt_with_metadata.prompt, output.completion(), debug_log
        )
        return SingleChunkQaOutput(
            answer=self._no_answer_to_none(output.completion().strip()),
            highlights=self._to_highlights(
                *self._extract_explanation_and_range(prompt_with_metadata, explanation),
                debug_log,
            ),
            debug_log=debug_log,
        )

    def _extract_explanation_and_range(
        self,
        prompt_with_metadata: PromptWithMetadata,
        explanation_response: ExplanationResponse,
    ) -> tuple[TextPromptItemExplanation, PromptRange]:
        item_index = 0
        prompt_text = prompt_with_metadata.prompt.items[item_index]
        assert isinstance(
            prompt_text, Text
        ), f"Expected `Text` prompt item, got {type(prompt_text)}."
        prompt_range = prompt_with_metadata.ranges["text"][item_index]
        text_prompt_item_explanation = explanation_response.explanations[0].items[
            item_index
        ]  # explanations[0], because one explanation for each target
        assert isinstance(text_prompt_item_explanation, TextPromptItemExplanation)
        return text_prompt_item_explanation.with_text(prompt_text), prompt_range

    def _no_answer_to_none(self, completion: str) -> Optional[str]:
        return completion if completion != self.NO_ANSWER_STR else None

    def _to_prompt_with_metadata(self, text: str, question: str) -> PromptWithMetadata:
        template = PromptTemplate(self.PROMPT_TEMPLATE_STR)
        return template.to_prompt_with_metadata(
            text=text, question=question, no_answer_text=self.NO_ANSWER_STR
        )

    def _complete(self, prompt: Prompt, debug_log: DebugLog) -> CompletionOutput:
        request = CompletionRequest(prompt)
        output = self.completion.run(CompletionInput(request=request, model=self.model))
        debug_log.debug("Completion", output.debug_log)
        debug_log.info(
            "Completion Input/Output",
            {"prompt": prompt, "completion": output.completion()},
        )
        return output

    def _explain(
        self, prompt: Prompt, target: str, debug_log: DebugLog
    ) -> ExplanationResponse:
        request = ExplanationRequest(prompt, target)
        response = self.client.explain(request, self.model)
        debug_log.debug(
            "Explanation Request/Response", {"request": request, "response": response}
        )
        return response

    def _to_highlights(
        self,
        text_prompt_item_explanation: TextPromptItemExplanation,
        prompt_range: PromptRange,
        debug_log: DebugLog,
    ) -> Sequence[str]:
        def prompt_range_contains_position(pos: int) -> bool:
            assert isinstance(prompt_range.start, TextCursor)
            assert isinstance(prompt_range.end, TextCursor)
            return prompt_range.start.position <= pos < prompt_range.end.position

        def prompt_range_overlaps_with_text_score(text_score: TextScoreWithRaw) -> bool:
            return prompt_range_contains_position(
                text_score.start
            ) or prompt_range_contains_position(text_score.start + text_score.length)

        overlapping = [
            text_score
            for text_score in text_prompt_item_explanation.scores
            if isinstance(text_score, TextScoreWithRaw)
            and prompt_range_overlaps_with_text_score(text_score)
        ]
        debug_log.info(
            "Explanation-Scores",
            [
                {
                    "text": text_score.text,
                    "score": text_score.score,
                }
                for text_score in overlapping
            ],
        )
        best_test_score = max(text_score.score for text_score in overlapping)
        return [
            text_score.text
            for text_score in overlapping
            if text_score.score == best_test_score
        ]

    def chop_highlight(self, text: str, score: TextScore, range: TextRange) -> str:
        start = score.start - range.start
        return text[max(0, start) : start + score.length]
