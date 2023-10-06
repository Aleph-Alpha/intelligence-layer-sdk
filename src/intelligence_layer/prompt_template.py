from collections import defaultdict
from dataclasses import dataclass
from re import finditer
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NewType,
    Sequence,
    Tuple,
    Union,
)
from uuid import UUID, uuid4
from liquid import Template

from aleph_alpha_client.prompt import Image, Prompt, PromptItem, Text, Tokens

Placeholder = NewType("Placeholder", UUID)


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
    positions: Mapping[str, Sequence[Union[TextPosition, PromptItemPosition]]]


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
        self.template = Template(template_str)
        self.placeholders: Dict[Placeholder, Union[Image, Tokens, str]] = {}

    def placeholder(self, value: Union[Image, Tokens, str]) -> Placeholder:
        """Saves a non-text prompt item to the template and returns a placeholder

        The placeholder is used to embed the prompt item in the template
        """
        id = Placeholder(uuid4())
        self.placeholders[id] = value
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
        liquid_prompt: str = self.template.render(**kwargs)
        template_variable_by_placeholder = {
            placeholder: variable_name
            for variable_name, placeholder in kwargs.items()
            if isinstance(placeholder, UUID) and placeholder in self.placeholders
        }
        placeholder_indices = self._compute_indices(
            self.placeholders.keys(), liquid_prompt
        )
        positions_by_placeholder: Dict[
            Placeholder, List[Union[TextPosition, PromptItemPosition]]
        ] = defaultdict(list)
        modalities = self._modalities_from(
            placeholder_indices, positions_by_placeholder, liquid_prompt
        )
        result = PromptData(
            Prompt(list(modalities)),
            {
                # template_variable_by_placeholder.get(placeholder) cannot be None as None-s are filtered
                template_variable_by_placeholder.get(placeholder): positions  # type: ignore
                for placeholder, positions in positions_by_placeholder.items()
                if template_variable_by_placeholder.get(placeholder)
            },
        )
        self.placeholders = {}
        return result

    def to_prompt(self, **kwargs: Any) -> Prompt:
        """Creates a `Prompt` from the template string and the given parameters.

        Provided parameters are passed to `liquid.Template.render`.
        """
        return self.to_prompt_data(**kwargs).prompt

    def _compute_indices(
        self, placeholders: Iterable[Placeholder], template: str
    ) -> Iterable[Tuple[int, int]]:
        if not self.placeholders:
            return []
        pattern = f"({'|'.join(str(placeholder) for placeholder in placeholders)})"
        return ((match.start(), match.end()) for match in finditer(pattern, template))

    def _modalities_from(
        self,
        placeholder_indices: Iterable[Tuple[int, int]],
        positions_by_placeholder: Dict[
            Placeholder, List[Union[TextPosition, PromptItemPosition]]
        ],
        template: str,
    ) -> Iterable[PromptItem]:
        last_to = 0
        accumulated_text = ""
        item_cnt = 0

        def new_prompt_item(item: PromptItem) -> PromptItem:
            nonlocal item_cnt, accumulated_text
            item_cnt += 1
            accumulated_text = ""
            return item

        def current_text_position(value: str) -> Iterable[TextPosition]:
            nonlocal item_cnt, accumulated_text
            yield TextPosition(
                item=item_cnt,
                start=len(accumulated_text),
                length=len(value),
            )
            accumulated_text += value

        for placeholder_from, placeholder_to in placeholder_indices:
            accumulated_text += template[last_to:placeholder_from]
            placeholder = Placeholder(UUID(template[placeholder_from:placeholder_to]))
            placeholder_value = self.placeholders[placeholder]
            if isinstance(placeholder_value, (Tokens, Image)):
                if accumulated_text:
                    yield new_prompt_item(Text.from_text(accumulated_text))
                positions_by_placeholder[placeholder].append(
                    PromptItemPosition(item=item_cnt)
                )
                yield new_prompt_item(placeholder_value)
            else:
                positions_by_placeholder[placeholder].extend(
                    current_text_position(placeholder_value)
                )
            last_to = placeholder_to
        if last_to < len(template):
            yield Text.from_text(accumulated_text + template[last_to:])
