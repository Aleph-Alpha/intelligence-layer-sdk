from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from re import finditer
from sys import intern
from typing import (
    Any,
    NewType,
    Optional,
    TextIO,
    Union,
)
from uuid import UUID, uuid4

from aleph_alpha_client.prompt import Image, Prompt, PromptItem, Text, Tokens
from liquid import BoundTemplate, Context, Environment
from liquid.ast import BlockNode, Node
from liquid.context import Namespace
from liquid.exceptions import LiquidTypeError
from liquid.expressions.common import parse_unchained_identifier
from liquid.expressions.filtered.lex import tokenize
from liquid.expressions.stream import TokenStream as AstTokenStream
from liquid.parse import expect, get_parser
from liquid.stream import TokenStream
from liquid.tag import Tag
from liquid.token import TOKEN_EOF, TOKEN_EXPRESSION, TOKEN_TAG

Placeholder = NewType("Placeholder", UUID)


@dataclass(frozen=True)
class TextCursor:
    """Defines a position with a `Text` prompt item.

    Args:
        item: the index of the prompt item within the `Prompt`
        position: the character position in the text of the item.

    Example:
    >>> from aleph_alpha_client import Prompt
    >>> from intelligence_layer.core import TextCursor
    >>> prompt = Prompt.from_text("This is a text")
    >>> # This denotes the "i" in "is" in the text-item of the `Prompt` above
    >>> cursor = TextCursor(item=0, position=5)
    """

    item: int
    position: int


@dataclass(frozen=True)
class PromptItemCursor:
    """Defines a position with a non-`Text` prompt item.

    Args:
        item: the index of the prompt item within the `Prompt`
    """

    item: int


Cursor = Union[TextCursor, PromptItemCursor]


@dataclass
class PromptRange:
    """Defines a range within a `Prompt`."""

    start: Cursor
    end: Cursor


@dataclass
class RichPrompt(Prompt):
    """The `Prompt` along with some metadata generated when a `PromptTemplate` is turned into a `Prompt`.

    Args:
      ranges: A mapping of range name to a `Sequence` of corresponding `PromptRange` instances.
    """

    ranges: Mapping[str, Sequence[PromptRange]] = field(default_factory=dict)


PROMPT_RANGE_TAG = intern("promptrange")
PROMPT_RANGE_END_TAG = intern("endpromptrange")


class PromptRangeTag(Tag):
    """Defines the liquid tag for the promptrange."""

    name = PROMPT_RANGE_TAG
    end = PROMPT_RANGE_END_TAG

    def __init__(self, env: Environment):
        super().__init__(env)
        self.parser = get_parser(env)

    def parse(self, stream: TokenStream) -> Node:
        expect(stream, TOKEN_TAG, PROMPT_RANGE_TAG)
        stream.next_token()
        expect(stream, TOKEN_EXPRESSION)

        name = str(
            parse_unchained_identifier(AstTokenStream(tokenize(stream.current.value)))
        )
        stream.next_token()
        block = self.parser.parse_block(stream, (PROMPT_RANGE_END_TAG, TOKEN_EOF))
        expect(stream, TOKEN_TAG, value=PROMPT_RANGE_END_TAG)
        return PromptRangeNode(block, name)


class PromptRangeContext(Context):
    """A liquid `Context` with some additional state used by the `PromptRangeNode`."""

    def __init__(
        self,
        env: Environment,
        globals: Optional[Namespace] = None,
        disabled_tags: Optional[list[str]] = None,
        copy_depth: int = 0,
        parent_context: Optional[Context] = None,
        loop_iteration_carry: int = 1,
        local_namespace_size_carry: int = 0,
        template: Optional[BoundTemplate] = None,
    ):
        super().__init__(
            env,
            globals,
            disabled_tags,
            copy_depth,
            parent_context,
            loop_iteration_carry,
            local_namespace_size_carry,
            template,
        )
        self._placeholder_range_names: dict[Placeholder, str] = {}

    def add_placeholder_range(self, placeholder: Placeholder, name: str) -> None:
        self._placeholder_range_names[placeholder] = name

    def placeholder_range_names(self) -> Mapping[Placeholder, str]:
        return self._placeholder_range_names


class PromptRangeNode(Node):
    """A liquid `Node` representing a promptrange."""

    def __init__(self, inner: BlockNode, name: str) -> None:
        super().__init__()
        self.inner = inner
        self.name = name
        self.placeholder = Placeholder(uuid4())

    def render_to_output(self, context: Context, buffer: TextIO) -> Optional[bool]:
        if not isinstance(context, PromptRangeContext):
            raise LiquidTypeError(
                f"Context not of expected type: {PromptRangeContext} (is: {type(context)})"
            )
        context.add_placeholder_range(self.placeholder, self.name)
        buffer.write(str(self.placeholder))
        self.inner.render(context, buffer)
        buffer.write(str(self.placeholder))
        return True


class PromptTemplate:
    """Allows to build a `Prompt` using the `liquid template language <https://shopify.github.io/liquid/>`_.

    To add non-text prompt items first you have to save it to the template with the `template.placeholder()` function.
    To embed the items in the template, pass the placeholder in the place(s) where you would like the items.

    Example:
        >>> from aleph_alpha_client import CompletionRequest, Tokens

        >>> from intelligence_layer.core import PromptTemplate

        >>> tokens = Tokens.from_token_ids([1, 2, 3])
        >>> template = PromptTemplate(
        ...     '''{%- for name in names -%}
        ...     Hello {{name}}!
        ...     {% endfor -%}
        ...     {{ image }}
        ...     ''')
        >>> placeholder = template.placeholder(tokens)
        >>> names = ["World", "Rutger"]
        >>> prompt = template.to_rich_prompt(names=names, image=placeholder)
        >>> request = CompletionRequest(prompt=prompt)
    """

    def __init__(self, template_str: str) -> None:
        """Initialize with the liquid template string.

        The template supports the custom liquid tag `promptrange`. This can be used to determine ranges
        within the `Prompt` primarily for downstream explainability tasks.

        Args:
            template_str: the liquid template string

        Example:
            >>> from intelligence_layer.core import PromptTemplate


            >>> template = PromptTemplate(
            ... '''Answer the following question given the input.
            ...
            ... Input: {% promptrange input %}{{text}}{% endpromptrange %}
            ... Question: {% promptrange question %}{{question}}{% endpromptrange %}
            ... Answer:''')
            >>> prompt_data = template.to_rich_prompt(text="Some text...", question="A question ...")
            >>> input_range = prompt_data.ranges.get("input")
        """
        env = Environment()
        env.add_tag(PromptRangeTag)
        self._template = env.from_string(template_str)
        self._prompt_item_placeholders: dict[Placeholder, Union[Image, Tokens]] = {}

    def placeholder(self, value: Union[Image, Tokens]) -> Placeholder:
        """Saves a non-text prompt item to the template and returns a placeholder.

        Args:
            value: Tokens to store
        The placeholder is used to embed the prompt item in the template
        """
        id = Placeholder(uuid4())
        self._prompt_item_placeholders[id] = value
        return id

    def _join_character(
        self, first_item: Union[Text, Image, Tokens, None], second_item: Text
    ) -> str:
        if (
            isinstance(first_item, Text)
            and not first_item.text[-1].isspace()
            and not second_item.text[0].isspace()
        ):
            return " "
        else:
            return ""

    def embed_prompt(self, prompt: Prompt) -> str:
        r"""Embeds a prompt in a prompt template.

        Adds whitespace between text items if there is no whitespace between them.
        In case of non-text prompt items, this embeds them into the end result.

        Args:
            prompt: prompt to embed in the template

        Example:
            >>> from aleph_alpha_client import Prompt, Text, Tokens

            >>> from intelligence_layer.core import PromptTemplate

            >>> user_prompt = Prompt([
            ... Tokens.from_token_ids([1, 2, 3]),
            ... Text.from_text("cool"),
            ... ])
            >>> template = PromptTemplate("Question: {{user_prompt}}\n Answer: ")
            >>> prompt = template.to_rich_prompt(user_prompt=template.embed_prompt(user_prompt))
        """
        prompt_text = ""
        last_item = None
        for item in prompt.items:
            if isinstance(item, Text):
                if len(item.text) == 0:
                    continue
                prompt_text = str.join(
                    self._join_character(last_item, item), [prompt_text, item.text]
                )
            else:
                prompt_text = str.join("", [prompt_text, str(self.placeholder(item))])
            last_item = item
        return prompt_text

    def to_rich_prompt(self, **kwargs: Any) -> RichPrompt:
        """Creates a `Prompt` along with metadata from the template string and the given parameters.

        Args:
             **kwargs: Parameters to enrich prompt with
         Currently, the only metadata returned is information about ranges that are marked in the template.
         Provided parameters are passed to `liquid.Template.render`.
        """
        context = PromptRangeContext(
            self._template.env,
            globals=self._template.make_globals(kwargs),
            template=self._template,
        )
        buffer = self._template._get_buffer()
        self._template.render_with_context(context, buffer, **kwargs)
        liquid_prompt = buffer.getvalue()
        placeholder_indices = self._compute_indices(
            chain(
                self._prompt_item_placeholders.keys(),
                context.placeholder_range_names().keys(),
            ),
            liquid_prompt,
        )
        modalities, placeholder_ranges = self._compute_modalities_and_ranges(
            placeholder_indices, context.placeholder_range_names(), liquid_prompt
        )

        result = RichPrompt(modalities, placeholder_ranges)
        self._reset_placeholder_state()
        return result

    def _reset_placeholder_state(self) -> None:
        self._prompt_item_placeholders = {}

    def _compute_indices(
        self, placeholders: Iterable[Placeholder], template: str
    ) -> Iterable[tuple[int, int]]:
        pattern = "|".join(str(placeholder) for placeholder in placeholders)
        return (
            (
                (match.start(), match.end())
                for match in finditer(f"({pattern})", template)
            )
            if pattern
            else []
        )

    def _compute_modalities_and_ranges(
        self,
        placeholder_indices: Iterable[tuple[int, int]],
        placeholder_range_names: Mapping[Placeholder, str],
        template: str,
    ) -> tuple[Sequence[PromptItem], Mapping[str, Sequence[PromptRange]]]:
        placeholder_ranges: dict[Placeholder, list[PromptRange]] = defaultdict(list)
        modalities = list(
            self._modalities_from(placeholder_indices, placeholder_ranges, template)
        )
        self._replace_start_cursors_of_non_text_items(modalities, placeholder_ranges)
        return modalities, {
            placeholder_range_names[placeholder]: ranges
            for placeholder, ranges in placeholder_ranges.items()
            if placeholder_range_names.get(placeholder)
        }

    @staticmethod
    def _replace_start_cursors_of_non_text_items(
        modalities: Sequence[PromptItem],
        placeholder_ranges: dict[Placeholder, list[PromptRange]],
    ) -> None:
        for prompt_ranges in placeholder_ranges.values():
            for index, range in enumerate(prompt_ranges):
                if not isinstance(modalities[range.start.item], Text):
                    prompt_ranges[index] = replace(
                        range, start=PromptItemCursor(range.start.item)
                    )

    def _modalities_from(
        self,
        placeholder_indices: Iterable[tuple[int, int]],
        placeholder_ranges: dict[Placeholder, list[PromptRange]],
        template: str,
    ) -> Iterable[PromptItem]:
        last_to = 0
        accumulated_text = ""
        item_cnt = 0
        range_starts: dict[Placeholder, TextCursor] = {}

        def new_prompt_item(item: PromptItem) -> PromptItem:
            nonlocal item_cnt, accumulated_text
            item_cnt += 1
            accumulated_text = ""
            return item

        def initial_start_text_cursor() -> TextCursor:
            return TextCursor(item=item_cnt, position=len(accumulated_text))

        def end_cursor() -> Cursor:
            return (
                TextCursor(item=item_cnt, position=len(accumulated_text))
                if accumulated_text
                else PromptItemCursor(item_cnt - 1)
            )

        def valid_range_for(
            placeholder: Placeholder, end: Cursor
        ) -> Iterable[PromptRange]:
            if end.item >= range_starts[placeholder].item:
                yield PromptRange(start=range_starts[placeholder], end=end)
            del range_starts[placeholder]

        for placeholder_from, placeholder_to in placeholder_indices:
            placeholder = Placeholder(UUID(template[placeholder_from:placeholder_to]))
            accumulated_text += template[last_to:placeholder_from]
            placeholder_prompt_item = self._prompt_item_placeholders.get(placeholder)
            if placeholder_prompt_item:
                if accumulated_text:
                    yield new_prompt_item(Text.from_text(accumulated_text))

                yield new_prompt_item(placeholder_prompt_item)
            else:
                if range_starts.get(placeholder):
                    placeholder_ranges[placeholder].extend(
                        valid_range_for(placeholder, end_cursor())
                    )
                else:
                    range_starts[placeholder] = initial_start_text_cursor()
            last_to = placeholder_to
        if last_to < len(template) or accumulated_text:
            yield Text.from_text(accumulated_text + template[last_to:])
