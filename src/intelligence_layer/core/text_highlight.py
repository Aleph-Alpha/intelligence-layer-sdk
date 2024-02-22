from typing import Iterable, Sequence

from aleph_alpha_client import (
    Prompt,
    PromptGranularity,
    Text,
    TextPromptItemExplanation,
)
from aleph_alpha_client.explanation import TextScoreWithRaw
from pydantic import BaseModel

from intelligence_layer.core.model import ControlModel, ExplainInput, ExplainOutput
from intelligence_layer.core.prompt_template import (
    Cursor,
    PromptRange,
    RichPrompt,
    TextCursor,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan


class TextHighlightInput(BaseModel):
    """The input for a text highlighting task.

    Attributes:
        rich_prompt: From client's PromptTemplate. Includes both the actual 'Prompt' as well as text range information.
            Supports liquid-template-language-style {% promptrange range_name %}/{% endpromptrange %} for range.
        target: The target that should be explained. Expected to follow the prompt.
        focus_ranges: The ranges contained in `rich_prompt` the returned highlights stem from. That means that each returned
            highlight overlaps with at least one character with one of the ranges listed here.
            If this set is empty highlights of the entire prompt are returned.
    """

    rich_prompt: RichPrompt
    target: str
    focus_ranges: frozenset[str] = frozenset()


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
    A range can be provided via use of the liquid language (see the example).
    In this case, the highlights will only refer to text within this range.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Example:
        >>> import os

        >>> from intelligence_layer.connectors import LimitedConcurrencyClient
        >>> from intelligence_layer.core import (
        ... 	InMemoryTracer,
        ... 	PromptTemplate,
        ... 	TextHighlight,
        ... 	TextHighlightInput,
        ... )

        >>> client = LimitedConcurrencyClient.from_token(os.getenv("AA_TOKEN"))
        >>> text_highlight = TextHighlight(client=client)
        >>> prompt_template_str = (
        ...		"{% promptrange r1 %}Question: What is 2 + 2?{% endpromptrange %}\\nAnswer:"
        ...	)
        >>> template = PromptTemplate(prompt_template_str)
        >>> rich_prompt = template.to_rich_prompt()
        >>> completion = " 4."
        >>> model = "luminous-base"
        >>> input = TextHighlightInput(
        ...	    rich_prompt=rich_prompt, target=completion, model=model
        ... )
        >>> output = text_highlight.run(input, InMemoryTracer())
    """

    def __init__(
        self,
        model: ControlModel,
        granularity: PromptGranularity = PromptGranularity.Sentence,
    ) -> None:
        super().__init__()
        self._model = model
        self._granularity = granularity

    def do_run(
        self, input: TextHighlightInput, task_span: TaskSpan
    ) -> TextHighlightOutput:
        self._raise_on_invalid_focus_range(input)
        explanation = self._explain(
            prompt=input.rich_prompt,
            target=input.target,
            task_span=task_span,
        )
        prompt_ranges = self._flatten_prompt_ranges(
            range
            for name, range in input.rich_prompt.ranges.items()
            if name in input.focus_ranges
        )
        text_prompt_item_explanations_and_indices = (
            self._extract_text_prompt_item_explanations_and_item_index(
                input.rich_prompt, explanation
            )
        )
        highlights = self._to_highlights(
            prompt_ranges,
            text_prompt_item_explanations_and_indices,
            task_span,
        )
        return TextHighlightOutput(highlights=highlights)

    def _raise_on_invalid_focus_range(self, input: TextHighlightInput) -> None:
        unknown_focus_ranges = input.focus_ranges - set(input.rich_prompt.ranges.keys())
        if unknown_focus_ranges:
            raise ValueError(f"Unknown focus ranges: {', '.join(unknown_focus_ranges)}")

    def _explain(
        self, prompt: Prompt, target: str, task_span: TaskSpan
    ) -> ExplainOutput:
        input = ExplainInput(
            prompt=prompt,
            target=target,
            prompt_granularity=self._granularity,
        )
        output = self._model.explain(input, task_span)
        return output

    def _flatten_prompt_ranges(
        self, prompt_ranges: Iterable[Sequence[PromptRange]]
    ) -> Sequence[PromptRange]:
        return [pr for prs in prompt_ranges for pr in prs]

    def _extract_text_prompt_item_explanations_and_item_index(
        self,
        prompt: Prompt,
        explain_output: ExplainOutput,
    ) -> Sequence[tuple[TextPromptItemExplanation, int]]:
        prompt_texts_and_indices = [
            (prompt_text, idx)
            for idx, prompt_text in enumerate(prompt.items)
            if isinstance(prompt_text, Text)
        ]
        text_prompt_item_explanations = [
            explanation
            for explanation in explain_output.explanations[0].items
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
        task_span: TaskSpan,
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
        task_span.log(
            "Raw explanation scores",
            [
                {
                    "text": text_score.text,
                    "score": text_score.score,
                }
                for text_score in overlapping_and_flat
            ],
        )
        if not overlapping_and_flat:
            return []
        z_scores = self._z_scores([s.score for s in overlapping_and_flat])
        scored_highlights = [
            ScoredTextHighlight(text=text_score.text, score=z_score)
            for text_score, z_score in zip(overlapping_and_flat, z_scores)
        ]
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
        return (
            cls._is_within_prompt_range(
                prompt_range,
                explanation_item_idx,
                text_score.start,
            )
            or cls._is_within_prompt_range(
                prompt_range,
                explanation_item_idx,
                text_score.start + text_score.length - 1,
            )
            or cls._is_within_text_score(
                text_score, explanation_item_idx, prompt_range.start
            )
        )

    @staticmethod
    def _is_within_text_score(
        text_score: TextScoreWithRaw,
        text_score_item: int,
        prompt_range_cursor: Cursor,
    ) -> bool:
        if text_score_item != prompt_range_cursor.item:
            return False
        assert isinstance(prompt_range_cursor, TextCursor)
        return (
            text_score.start
            <= prompt_range_cursor.position
            <= text_score.start + text_score.length - 1
        )

    @staticmethod
    def _is_within_prompt_range(
        prompt_range: PromptRange,
        item_check: int,
        pos_check: int,
    ) -> bool:
        if item_check < prompt_range.start.item or item_check > prompt_range.end.item:
            return False
        if item_check == prompt_range.start.item:
            # must be a text cursor, because has same index as TextScoreWithRaw
            assert isinstance(prompt_range.start, TextCursor)
            if pos_check < prompt_range.start.position:
                return False
        if item_check == prompt_range.end.item:
            assert isinstance(prompt_range.end, TextCursor)  # see above
            if pos_check > prompt_range.end.position:
                return False
        return True

    @staticmethod
    def _z_scores(data: Sequence[float]) -> Sequence[float]:
        mean = 0  # assuming a mean of 0 (population mean), therefore also assuming n instead of n-1 (population df)
        stdev = (
            (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
            if len(data) > 1
            else 0
        )
        return [((x - mean) / stdev if stdev > 0 else 0) for x in data]

    def _filter_highlights(
        self,
        scored_highlights: Sequence[ScoredTextHighlight],
        z_score_limit: float = 1.0,
    ) -> Sequence[ScoredTextHighlight]:
        return [h for h in scored_highlights if abs(h.score) >= z_score_limit]
