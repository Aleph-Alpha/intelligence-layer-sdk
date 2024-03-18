import itertools
from typing import Iterable, Mapping, Sequence

from aleph_alpha_client import (
    Prompt,
    PromptGranularity,
    TextPromptItemExplanation,
    TextScore,
)
from pydantic import BaseModel

from intelligence_layer.core.model import AlephAlphaModel, ExplainInput, ExplainOutput
from intelligence_layer.core.prompt_template import (
    Cursor,
    PromptRange,
    RichPrompt,
    TextCursor,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer.tracer import TaskSpan


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
        start: The start index of the highlight.
        end: The end index of the highlight.
        score: The score of the highlight. Normalized to be between zero and one, with higher being more important.
    """

    start: int
    end: int
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
        model: The model used throughout the task for model related API calls.
        granularity: At which granularity should the target be explained in terms of the prompt.
        threshold: After normalization, everything highlight below this value will be dropped.
        clamp: Control whether highlights should be clamped to a focus range if they intersect it.

    Example:
        >>> import os

        >>> from intelligence_layer.core import (
        ... 	InMemoryTracer,
        ... 	PromptTemplate,
        ... 	TextHighlight,
        ... 	TextHighlightInput,
        ...     AlephAlphaModel
        ... )

        >>> model = AlephAlphaModel(name="luminous-base")
        >>> text_highlight = TextHighlight(model=model)
        >>> prompt_template_str = (
        ...		"{% promptrange r1 %}Question: What is 2 + 2?{% endpromptrange %}\\nAnswer:"
        ...	)
        >>> template = PromptTemplate(prompt_template_str)
        >>> rich_prompt = template.to_rich_prompt()
        >>> completion = " 4."
        >>> model = "luminous-base"
        >>> input = TextHighlightInput(
        ...	    rich_prompt=rich_prompt, target=completion, focus_ranges=frozenset({"r1"})
        ... )
        >>> output = text_highlight.run(input, InMemoryTracer())
    """

    def __init__(
        self,
        model: AlephAlphaModel,
        granularity: PromptGranularity | None = None,
        threshold: float = 0.1,
        clamp: bool = False,
    ) -> None:
        super().__init__()
        self._threshold = threshold
        self._model = model
        self._granularity = granularity
        self._clamp_to_focus = clamp

    def do_run(
        self, input: TextHighlightInput, task_span: TaskSpan
    ) -> TextHighlightOutput:
        self._raise_on_invalid_focus_range(input)
        explanation = self._explain(
            prompt=input.rich_prompt,
            target=input.target,
            task_span=task_span,
        )
        prompt_ranges = self._filter_and_flatten_prompt_ranges(
            input.focus_ranges, input.rich_prompt.ranges
        )

        text_prompt_item_explanations_and_indices = (
            self._extract_text_prompt_item_explanations_and_item_index(explanation)
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

    def _filter_and_flatten_prompt_ranges(
        self,
        focus_ranges: frozenset[str],
        input_ranges: Mapping[str, Sequence[PromptRange]],
    ) -> Sequence[PromptRange]:
        relevant_ranges = (
            range for name, range in input_ranges.items() if name in focus_ranges
        )
        return list(itertools.chain.from_iterable(relevant_ranges))

    def _extract_text_prompt_item_explanations_and_item_index(
        self,
        explain_output: ExplainOutput,
    ) -> Iterable[tuple[TextPromptItemExplanation, int]]:
        return (
            (explanation, index)
            for index, explanation in enumerate(explain_output.explanations[0].items)
            if isinstance(explanation, TextPromptItemExplanation)
        )

    def _to_highlights(
        self,
        prompt_ranges: Sequence[PromptRange],
        text_prompt_item_explanations_and_indices: Iterable[
            tuple[TextPromptItemExplanation, int]
        ],
        task_span: TaskSpan,
    ) -> Sequence[ScoredTextHighlight]:
        relevant_text_scores: list[TextScore] = []
        for (
            text_prompt_item_explanation,
            explanation_idx,
        ) in text_prompt_item_explanations_and_indices:
            for text_score in text_prompt_item_explanation.scores:
                assert isinstance(text_score, TextScore)  # for typing

                if self._is_relevant_explanation(
                    explanation_idx, text_score, prompt_ranges
                ):
                    relevant_text_scores.append(text_score)

        task_span.log(
            "Raw explanation scores",
            [
                {
                    "start": text_score.start,
                    "end": text_score.start + text_score.length,
                    "score": text_score.score,
                }
                for text_score in relevant_text_scores
            ],
        )

        relevant_text_scores = self._clamp_ranges_to_focus(
            prompt_ranges, relevant_text_scores
        )

        text_highlights = [
            ScoredTextHighlight(
                start=text_score.start,
                end=text_score.start + text_score.length,
                score=text_score.score,
            )
            for text_score in relevant_text_scores
        ]

        return self._normalize_and_filter(text_highlights)

    def _clamp_ranges_to_focus(
        self,
        prompt_ranges: Sequence[PromptRange],
        relevant_highlights: Sequence[TextScore],
    ) -> Sequence[TextScore]:
        # TODO we just assume that its all text for now, but thats not a given. This will need to be replaced
        for prompt_range in prompt_ranges:
            assert isinstance(prompt_range.start, TextCursor)
            assert isinstance(prompt_range.end, TextCursor)

        if not self._should_clamp(prompt_ranges):
            return relevant_highlights

        new_relevant_text_scores: Sequence[TextScore] = []
        for highlight in relevant_highlights:

            def _get_overlap(range: PromptRange) -> int:
                return min(
                    highlight.start + highlight.length, range.end.position
                ) - max(highlight.start, range.start.position)

            most_overlapping_range = sorted(
                prompt_ranges,
                key=_get_overlap,
            )[-1]
            new_relevant_text_scores.append(
                self._clamp_to_range(highlight, most_overlapping_range)
            )
        return new_relevant_text_scores

    def _should_clamp(self, prompt_ranges: Sequence[PromptRange]) -> bool:
        def are_overlapping(p1: PromptRange, p2: PromptRange) -> bool:
            if p1.start.position > p2.start.position:
                p1, p2 = p2, p1
            return (
                p1.start.position <= p2.end.position
                and p1.end.position >= p2.start.position
            )

        if not self._clamp_to_focus or len(prompt_ranges) == 0:
            return False

        # this check is relatively expensive if no ranges are overlapping
        has_overlapping_ranges = any(
            are_overlapping(p1, p2)
            for p1, p2 in itertools.permutations(prompt_ranges, 2)
        )
        if has_overlapping_ranges:
            print(
                "TextHighlighting with clamping is on, but focus ranges are overlapping. Disabling clamping."
            )
        return not has_overlapping_ranges

    def _clamp_to_range(
        self, highlight: TextScore, focus_range: PromptRange
    ) -> TextScore:
        new_start = highlight.start
        new_length = highlight.length
        new_score = highlight.score
        cut_characters = 0
        # clamp start
        if highlight.start < focus_range.start.position:
            cut_characters = focus_range.start.position - highlight.start
            new_start = focus_range.start.position
            new_length -= cut_characters
        # clamp end
        if highlight.start + highlight.length > focus_range.end.position:
            n_cut_at_end = (highlight.start + highlight.length) - focus_range.end.position
            cut_characters += n_cut_at_end
            new_length -= n_cut_at_end

        if cut_characters:
            new_score = highlight.score * (new_length / highlight.length)
        return TextScore(new_start, new_length, new_score)

    def _normalize_and_filter(
        self, text_highlights: Sequence[ScoredTextHighlight]
    ) -> Sequence[ScoredTextHighlight]:
        max_score = max(highlight.score for highlight in text_highlights)
        divider = max(
            1, max_score
        )  # We only normalize if the max score is above a threshold to avoid noisy attribution in case where

        for highlight in text_highlights:
            highlight.score = max(highlight.score / divider, 0)

        return [
            highlight
            for highlight in text_highlights
            if highlight.score >= self._threshold
        ]

    def _is_relevant_explanation(
        self,
        explanation_idx: int,
        text_score: TextScore,
        prompt_ranges: Iterable[PromptRange],
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
        text_score: TextScore,
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
        text_score: TextScore,
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
            # must be a text cursor, because has same index as TextScore
            assert isinstance(prompt_range.start, TextCursor)
            if pos_check < prompt_range.start.position:
                return False
        if item_check == prompt_range.end.item:
            assert isinstance(prompt_range.end, TextCursor)  # see above
            if pos_check > prompt_range.end.position:
                return False
        return True
