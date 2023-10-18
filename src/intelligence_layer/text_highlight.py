import statistics
from typing import Iterable, Sequence

from aleph_alpha_client import (
    Client,
    ExplanationRequest,
    ExplanationResponse,
    PromptGranularity,
    Text,
    TextPromptItemExplanation,
    Prompt,
)
from aleph_alpha_client.explanation import TextScoreWithRaw
from pydantic import BaseModel, Field

from intelligence_layer.prompt_template import (
    PromptRange,
    PromptWithMetadata,
    TextCursor,
)
from intelligence_layer.task import (
    DebugLogger,
    Task,
)


class TextHighlightInput(BaseModel):
    """The input for a text highlighting task.

    Attributes:
        prompt_with_metadata: From client's PromptTemplate. Includes both the actual 'Prompt' as well as text range information.
            Supports liquid-template-language-style {% promptrange range_name %}/{% endpromptrange %} for range.
        target: The target that should be explained. Expected to follow the prompt.
        model: A valid Aleph Alpha model name.
        focus_ranges: The ranges contained in `prompt_with_metadata` the returned highlights stem from. That means that each returned
            highlight overlaps with at least one character with one of the ranges listed here.
    """

    prompt_with_metadata: PromptWithMetadata
    target: str
    model: str
    focus_ranges: set[str] = Field(default_factory=set)


class ScoredTextHighlight(BaseModel):
    """A substring of the input prompt scored for relevance with regard to the output.

    Attributes:
        text: The highlighted part of the prompt.
        score: The z-score of the highlight. Depicts relevance of this highlight in relation to all other highlights. Can be positive (support) or negative (contradiction).

    """

    text: str
    score: float


class TextHighlightOutput(BaseModel):
    """The output of a text highlighting task.

    Attributes:
        highlights: A sequence of 'ScoredTextHighlight's.

    """

    highlights: Sequence[ScoredTextHighlight]


class TextHighlight(Task[TextHighlightInput, TextHighlightOutput]):
    """Generates text highlights given a prompt and completion.

    For a given prompt and target (completion), extracts the parts of the prompt responsible for generation.

    A range can be provided in the input 'PromptWithMetadata' via use of the liquid language (see the example). In this case, the highlights will only refer to text within this range.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Example:
    >>> prompt_template_str = "{% promptrange r1 %}Question: What is 2 + 2?{% endpromptrange %}\nAnswer:"
    >>> template = PromptTemplate(prompt_template_str)
    >>> prompt_with_metadata = template.to_prompt_with_metadata()
    >>> completion = " 4."
    >>> model = "luminous-base"
    >>> input = TextHighlightInput(
    >>>     prompt_with_metadata=prompt_with_metadata, target=completion, model=model
    >>> )
    >>> output = text_highlight.run(input, InMemoryLogger(name="Highlight"))

    """

    _client: Client

    def __init__(self, client: Client) -> None:
        super().__init__()
        self._client = client

    def run(
        self, input: TextHighlightInput, logger: DebugLogger
    ) -> TextHighlightOutput:
        explanation = self._explain(
            prompt=input.prompt_with_metadata.prompt,
            target=input.target,
            model=input.model,
            logger=logger,
        )
        prompt_ranges = self._flatten_prompt_ranges(
            input.prompt_with_metadata.ranges.values()
        )
        text_prompt_item_explanations_and_indices = (
            self._extract_text_prompt_item_explanations_and_item_index(
                input.prompt_with_metadata.prompt, explanation
            )
        )
        highlights = self._to_highlights(
            prompt_ranges,
            text_prompt_item_explanations_and_indices,
            logger,
        )
        return TextHighlightOutput(highlights=highlights)

    def _explain(
        self, prompt: Prompt, target: str, model: str, logger: DebugLogger
    ) -> ExplanationResponse:
        request = ExplanationRequest(
            prompt,
            target,
            prompt_granularity=PromptGranularity.Sentence,
        )
        response = self._client.explain(request, model)
        logger.log(
            "Explanation Request/Response",
            {"request": request.to_json()},
        )
        return response

    def _flatten_prompt_ranges(
        self, prompt_ranges: Iterable[Sequence[PromptRange]]
    ) -> Sequence[PromptRange]:
        return [pr for prs in prompt_ranges for pr in prs]

    def _extract_text_prompt_item_explanations_and_item_index(
        self,
        prompt: Prompt,
        explanation_response: ExplanationResponse,
    ) -> Sequence[tuple[TextPromptItemExplanation, int]]:
        prompt_texts_and_indices = [
            (prompt_text, idx)
            for idx, prompt_text in enumerate(prompt.items)
            if isinstance(prompt_text, Text)
        ]
        text_prompt_item_explanations = [
            explanation
            for explanation in explanation_response.explanations[0].items
            if isinstance(explanation, TextPromptItemExplanation)
        ]  # explanations[0], because one explanation for each target
        assert len(prompt_texts_and_indices) == len(text_prompt_item_explanations)
        return [
            (
                text_prompt_item_explanation.with_text(prompt_text_and_index[0]),
                prompt_text_and_index[1],
            )
            for prompt_text_and_index, text_prompt_item_explanation in zip(
                prompt_texts_and_indices, text_prompt_item_explanations
            )
        ]

    def _to_highlights(
        self,
        prompt_ranges: Sequence[PromptRange],
        text_prompt_item_explanations_and_indices: Sequence[
            tuple[TextPromptItemExplanation, int]
        ],
        logger: DebugLogger,
    ) -> Sequence[ScoredTextHighlight]:
        overlapping_and_flat = [
            text_score
            for text_prompt_item_explanation, explanation_idx in text_prompt_item_explanations_and_indices
            for text_score in text_prompt_item_explanation.scores
            if isinstance(text_score, TextScoreWithRaw)
            and self._is_relevant_explanation(
                explanation_idx, text_score, prompt_ranges
            )
        ]
        logger.log(
            "Explanation scores",
            [
                {
                    "text": text_score.text,
                    "score": text_score.score,
                }
                for text_score in overlapping_and_flat
            ],
        )
        z_scores = self._z_scores([s.score for s in overlapping_and_flat], logger)
        scored_highlights = [
            ScoredTextHighlight(text=text_score.text, score=z_score)
            for text_score, z_score in zip(overlapping_and_flat, z_scores)
        ]
        logger.log("Unfiltered highlights", scored_highlights)
        return self._filter_highlights(scored_highlights)

    def _is_relevant_explanation(
        self,
        explanation_idx: int,
        text_score: TextScoreWithRaw,
        prompt_ranges: Sequence[PromptRange],
    ) -> bool:
        return (
            any(
                self._prompt_range_overlaps_with_text_score(
                    prompt_range, text_score, explanation_idx
                )
                for prompt_range in prompt_ranges
            )
            or not prompt_ranges
        )

    @classmethod
    def _prompt_range_overlaps_with_text_score(
        cls,
        prompt_range: PromptRange,
        text_score: TextScoreWithRaw,
        explanation_item_idx: int,
    ) -> bool:
        return cls._is_within_prompt_range(
            prompt_range,
            explanation_item_idx,
            text_score.start,
        ) or cls._is_within_prompt_range(
            prompt_range,
            explanation_item_idx,
            text_score.start + text_score.length - 1,
        )

    @staticmethod
    def _is_within_prompt_range(
        prompt_range: PromptRange,
        item_check: int,
        pos_check: int,
    ) -> bool:
        assert isinstance(prompt_range.start, TextCursor)
        assert isinstance(prompt_range.end, TextCursor)
        if item_check < prompt_range.start.item or item_check > prompt_range.end.item:
            return False
        elif (
            item_check == prompt_range.start.item
            and pos_check < prompt_range.start.position
        ):
            return False
        elif (
            item_check == prompt_range.end.item
            and pos_check > prompt_range.end.position
        ):
            return False
        return True

    @staticmethod
    def _z_scores(data: Sequence[float], logger: DebugLogger) -> Sequence[float]:
        mean = statistics.mean(data)
        stdev = (
            statistics.stdev(data) if len(data) > 1 else 0
        )  # standard deviation not defined for n < 2
        logger.log("Highlight statistics", {"mean": mean, "std_dev": stdev})
        return [((x - mean) / stdev if stdev > 0 else 0) for x in data]

    def _filter_highlights(
        self,
        scored_highlights: Sequence[ScoredTextHighlight],
        z_score_limit: float = 0.5,
    ) -> Sequence[ScoredTextHighlight]:
        return [h for h in scored_highlights if abs(h.score) >= z_score_limit]
