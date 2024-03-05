import itertools
from typing import Iterable, Mapping, Sequence

from aleph_alpha_client import (
    Prompt,
    PromptGranularity,
    Text,
    TextPromptItemExplanation,
    TextScore,
)
from aleph_alpha_client.explanation import TextScoreWithRaw
from pydantic import BaseModel

from intelligence_layer.core.model import (
    AlephAlphaModel,
    ExplainInput,
    ExplainOutput,
    LuminousControlModel,
)
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
        score: The z-score of the highlight. Depicts relevance of this highlight in relation to all other highlights. Can be positive (support) or negative (contradiction).
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
        model: AlephAlphaModel | None = None,
        granularity: PromptGranularity | None = None,
    ) -> None:
        super().__init__()
        self._model = model or LuminousControlModel()
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
        prompt_ranges = self._filter_and_flatten_prompt_ranges(
            input.focus_ranges, input.rich_prompt.ranges
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

    def _filter_and_flatten_prompt_ranges(
        self,
        focus_ranges: frozenset[str],
        input_ranges: Mapping[str, Sequence[PromptRange]],
    ) -> Iterable[PromptRange]:
        relevant_ranges = (
            range for name, range in input_ranges.items() if name in focus_ranges
        )
        return itertools.chain.from_iterable(relevant_ranges)

    def _extract_text_prompt_item_explanations_and_item_index(
        self,
        prompt: Prompt,
        explain_output: ExplainOutput,
    ) -> Iterable[tuple[TextPromptItemExplanation, int]]:
        return (
            (explanation, index)
            for index, explanation in enumerate(explain_output.explanations[0].items)
            if isinstance(explanation, TextPromptItemExplanation)
        )

    def _to_highlights(
        self,
        prompt_ranges: Iterable[PromptRange],
        text_prompt_item_explanations_and_indices: Iterable[
            tuple[TextPromptItemExplanation, int]
        ],
        task_span: TaskSpan,
    ) -> Sequence[ScoredTextHighlight]:
        overlapping_and_flat = []
        for (
            text_prompt_item_explanation,
            explanation_idx,
        ) in text_prompt_item_explanations_and_indices:
            for text_score in text_prompt_item_explanation.scores:
                assert isinstance(text_score, TextScore)
                if self._is_relevant_explanation(
                    explanation_idx, text_score, prompt_ranges
                ):
                    overlapping_and_flat.append(text_score)
        # task_span.log(
        #     "Raw explanation scores",
        #     [
        #         {
        #             "text": text_score.text,
        #             "score": text_score.score,
        #         }
        #         for text_score in overlapping_and_flat
        #     ],
        # ) We should not need this anymore, since we do not modify the scores anymore

        return [
            ScoredTextHighlight(
                start=text_score.start,
                end=text_score.start + text_score.length,
                score=text_score.score,
            )
            for text_score in overlapping_and_flat
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
            # must be a text cursor, because has same index as TextScoreWithRaw
            assert isinstance(prompt_range.start, TextCursor)
            if pos_check < prompt_range.start.position:
                return False
        if item_check == prompt_range.end.item:
            assert isinstance(prompt_range.end, TextCursor)  # see above
            if pos_check > prompt_range.end.position:
                return False
        return True
