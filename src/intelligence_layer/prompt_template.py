from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import chain
from re import finditer
from sys import intern
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NewType,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)
from uuid import UUID, uuid4
from liquid import BoundTemplate, Context, Environment, Template
from liquid.tag import Tag
from liquid.parse import get_parser, expect
from liquid.token import TOKEN_TAG, TOKEN_EOF, TOKEN_EXPRESSION
from liquid.ast import Node, BlockNode
from liquid.expressions.common import parse_unchained_identifier
from liquid.expressions.filtered.lex import tokenize
from liquid.stream import TokenStream
from liquid.expressions.stream import TokenStream as AstTokenStream
from liquid.exceptions import LiquidTypeError
from liquid.context import Namespace

from aleph_alpha_client.prompt import Image, Prompt, PromptItem, Text, Tokens

Placeholder = NewType("Placeholder", UUID)


@dataclass(frozen=True)
class TextCursor:
    item: int
    position: int


@dataclass(frozen=True)
class PromptItemCursor:
    item: int


@dataclass
class PromptRange:
    start: Union[TextCursor, PromptItemCursor]
    end: Union[TextCursor, PromptItemCursor]


@dataclass(frozen=True)
class TextPosition:
    item: int
    start: int
    length: int


@dataclass(frozen=True)
class PromptItemPosition:
    item: int


@dataclass(frozen=True)
class PromptData:
    prompt: Prompt
    ranges: Mapping[str, Sequence[PromptRange]]


PROMPT_RANGE_TAG = intern("promptrange")
PROMPT_RANGE_END_TAG = intern("endpromptrange")


class PromptRangeTag(Tag):
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
    def __init__(
        self,
        env: Environment,
        globals: Optional[Namespace] = None,
        disabled_tags: Optional[List[str]] = None,
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
    def __init__(self, inner: BlockNode, name: str) -> None:
        super().__init__()
        self.inner = inner
        self.name = name
        self.placeholder = Placeholder(uuid4())

    def render_to_output(self, context: Context, buffer: TextIO) -> Optional[bool]:
        if not isinstance(context, PromptRangeContext):
            raise LiquidTypeError("Context not of expected type")
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
        >>> image = Image.from_file(Path("path-to-image"))
        >>> template = PromptTemplate(
            '''{%- for name in names -%}
            Hello {{name}}!
            {% endfor -%}
            {{ image }}
            ''')
        >>> placeholder = template.placeholder(image)
        >>> names = ["World", "Rutger"]
        >>> prompt = template.to_prompt(names=names, image=placeholder)
        >>> request = CompletionRequest(prompt=prompt)
    """

    def __init__(self, template_str: str) -> None:
        """Initialize with the liquid template string.

        Parameters:
            template_str: the liquid template string
        """
        env = Environment()
        env.add_tag(PromptRangeTag)
        self.template = env.from_string(template_str)
        self.prompt_item_placeholders: Dict[Placeholder, Union[Image, Tokens]] = {}

    def placeholder(self, value: Union[Image, Tokens]) -> Placeholder:
        """Saves a non-text prompt item to the template and returns a placeholder

        The placeholder is used to embed the prompt item in the template
        """
        id = Placeholder(uuid4())
        self.prompt_item_placeholders[id] = value
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
        """Embeds a prompt in a prompt template

        Adds whitespace between text items if there is no whitespace between them.
        In case of non-text prompt items, this embeds them into the end result.

        Example:
            >>> user_prompt = Prompt(
                    [
                        Tokens.from_token_ids([1, 2, 3]),
                        Text.from_text("cool"),
                        Image.from_file(Path("path-to-image")),
                    ]
                )
            >>> template = PromptTemplate("Question: {{user_prompt}}\\n Answer: ")
            >>> prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

        Parameters:
            prompt: prompt to embed in the template
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

    def to_prompt_data(self, **kwargs: Any) -> PromptData:
        """Creates a `PromptData` from the template string and the given parameters.

        Provided parameters are passed to `liquid.Template.render`.
        """
        context = PromptRangeContext(
            self.template.env,
            globals=self.template.make_globals(kwargs),
            template=self.template,
        )
        buffer = self.template._get_buffer()
        self.template.render_with_context(context, buffer, **kwargs)
        liquid_prompt = buffer.getvalue()
        placeholder_indices = self._compute_indices(
            chain(
                self.prompt_item_placeholders.keys(),
                context.placeholder_range_names().keys(),
            ),
            liquid_prompt,
        )
        modalities, placeholder_ranges = self._compute_modalities_and_ranges(
            placeholder_indices, context.placeholder_range_names(), liquid_prompt
        )

        result = PromptData(Prompt(modalities), placeholder_ranges)
        self.prompt_item_placeholders = {}
        return result

    def to_prompt(self, **kwargs: Any) -> Prompt:
        """Creates a `Prompt` from the template string and the given parameters.

        Provided parameters are passed to `liquid.Template.render`.
        """
        return self.to_prompt_data(**kwargs).prompt

    def _compute_indices(
        self, placeholders: Iterable[Placeholder], template: str
    ) -> Iterable[Tuple[int, int]]:
        if not self.prompt_item_placeholders:
            return []
        pattern = f"({'|'.join(str(placeholder) for placeholder in placeholders)})"
        return ((match.start(), match.end()) for match in finditer(pattern, template))

    def _compute_modalities_and_ranges(
        self,
        placeholder_indices: Iterable[Tuple[int, int]],
        placeholder_range_names: Mapping[Placeholder, str],
        template: str,
    ) -> Tuple[Sequence[PromptItem], Mapping[str, Sequence[PromptRange]]]:
        placeholder_ranges: Dict[Placeholder, List[PromptRange]] = defaultdict(list)
        modalities = list(
            self._modalities_from(placeholder_indices, placeholder_ranges, template)
        )
        self.replace_start_cursors_of_non_text_items(modalities, placeholder_ranges)
        return modalities, {
            placeholder_range_names[placeholder]: ranges
            for placeholder, ranges in placeholder_ranges.items()
            if placeholder_range_names.get(placeholder)
        }

    @staticmethod
    def replace_start_cursors_of_non_text_items(
        modalities: Sequence[PromptItem],
        placeholder_ranges: Dict[Placeholder, List[PromptRange]],
    ) -> None:
        for prompt_ranges in placeholder_ranges.values():
            for index, range in enumerate(prompt_ranges):
                if not isinstance(modalities[range.start.item], Text):
                    prompt_ranges[index] = replace(
                        range, start=PromptItemCursor(range.start.item)
                    )

    def _modalities_from(
        self,
        placeholder_indices: Iterable[Tuple[int, int]],
        placeholder_ranges: dict[Placeholder, List[PromptRange]],
        template: str,
    ) -> Iterable[PromptItem]:
        last_to = 0
        accumulated_text = ""
        item_cnt = 0
        range_starts: Dict[Placeholder, Union[TextCursor, PromptItemCursor]] = {}

        def new_prompt_item(item: PromptItem) -> PromptItem:
            nonlocal item_cnt, accumulated_text
            item_cnt += 1
            accumulated_text = ""
            return item

        def initial_start_text_cursor() -> Union[TextCursor, PromptItemCursor]:
            return TextCursor(item=item_cnt, position=len(accumulated_text))

        def end_cursor() -> Union[TextCursor, PromptItemCursor]:
            return (
                TextCursor(item=item_cnt, position=len(accumulated_text))
                if accumulated_text
                else PromptItemCursor(item_cnt - 1)
            )

        for placeholder_from, placeholder_to in placeholder_indices:
            placeholder = Placeholder(UUID(template[placeholder_from:placeholder_to]))
            accumulated_text += template[last_to:placeholder_from]
            placeholder_prompt_item = self.prompt_item_placeholders.get(placeholder)
            if placeholder_prompt_item:
                if accumulated_text:
                    yield new_prompt_item(Text.from_text(accumulated_text))

                yield new_prompt_item(placeholder_prompt_item)
            else:
                if range_starts.get(placeholder):
                    placeholder_ranges[placeholder].append(
                        PromptRange(
                            start=range_starts[placeholder],
                            end=end_cursor(),
                        )
                    )
                    del range_starts[placeholder]
                else:
                    range_starts[placeholder] = initial_start_text_cursor()
            last_to = placeholder_to
        if last_to < len(template):
            yield Text.from_text(accumulated_text + template[last_to:])
