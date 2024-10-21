from pathlib import Path
from textwrap import dedent

from aleph_alpha_client import Prompt
from aleph_alpha_client.prompt import Image, PromptItem, Text, Tokens
from liquid.exceptions import LiquidSyntaxError, LiquidTypeError
from pytest import raises

from intelligence_layer.core.prompt_template import (
    PromptItemCursor,
    PromptRange,
    PromptTemplate,
    RichPrompt,
    TextCursor,
)


def rich_prompt_from_text(text: str) -> RichPrompt:
    return RichPrompt(items=[Text.from_text(text)])


def test_to_prompt_with_text_array() -> None:
    template = PromptTemplate(
        """
{%- for name in names -%}
Hello {{name}}!
{% endfor -%}
        """
    )
    names = ["World", "Rutger"]

    prompt = template.to_rich_prompt(names=names)

    expected = "".join([f"Hello {name}!\n" for name in names])
    assert prompt == rich_prompt_from_text(expected)


def test_to_prompt_with_invalid_input() -> None:
    template = PromptTemplate(
        """
{%- for name in names -%}
Hello {{name}}!
{% endfor -%}
        """
    )

    with raises(LiquidTypeError):
        template.to_rich_prompt(names=7)


def test_to_prompt_with_single_image(prompt_image: Image) -> None:
    template = PromptTemplate(
        """Some Text.
{{whatever}}
More Text
"""
    )

    prompt = template.to_rich_prompt(whatever=template.placeholder(prompt_image))

    expected = RichPrompt(
        [
            Text.from_text("Some Text.\n"),
            prompt_image,
            Text.from_text("\nMore Text\n"),
        ],
        ranges={},
    )
    assert prompt == expected


def test_to_prompt_with_image_sequence(prompt_image: Image) -> None:
    template = PromptTemplate(
        """
{%- for image in images -%}
{{image}}
{%- endfor -%}
        """
    )

    prompt = template.to_rich_prompt(
        images=[template.placeholder(prompt_image), template.placeholder(prompt_image)]
    )

    expected = RichPrompt([prompt_image, prompt_image])
    assert prompt == expected


def test_to_prompt_with_mixed_modality_variables(prompt_image: Image) -> None:
    template = PromptTemplate("""{{image}}{{name}}{{image}}""")

    prompt = template.to_rich_prompt(
        image=template.placeholder(prompt_image), name="whatever"
    )

    expected = RichPrompt([prompt_image, Text.from_text("whatever"), prompt_image])
    assert prompt == expected


def test_to_prompt_with_unused_image(prompt_image: Image) -> None:
    template = PromptTemplate("cool")

    prompt = template.to_rich_prompt(images=template.placeholder(prompt_image))

    assert prompt == rich_prompt_from_text("cool")
    assert prompt == RichPrompt(items=[Text("cool", controls=[])])


def test_to_prompt_with_multiple_different_images(prompt_image: Image) -> None:
    image_source_path = Path(__file__).parent.parent / "image_example.jpg"
    second_image = Image.from_file(image_source_path)

    template = PromptTemplate("""{{image_1}}{{image_2}}""")

    prompt = template.to_rich_prompt(
        image_1=template.placeholder(prompt_image),
        image_2=template.placeholder(second_image),
    )

    assert prompt == RichPrompt([prompt_image, second_image])


def test_to_prompt_with_embedded_prompt(prompt_image: Image) -> None:
    user_prompt = RichPrompt([Text.from_text("Cool"), prompt_image])

    template = PromptTemplate("""{{user_prompt}}""")

    prompt = template.to_rich_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_does_not_add_whitespace_after_image(prompt_image: Image) -> None:
    user_prompt = RichPrompt([prompt_image, Text.from_text("Cool"), prompt_image])

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_rich_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_skips_empty_strings() -> None:
    user_prompt = RichPrompt(
        [Text.from_text("Cool"), Text.from_text(""), Text.from_text("Also cool")]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_rich_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == RichPrompt([Text.from_text("Cool Also cool")])


def test_to_prompt_adds_whitespaces() -> None:
    user_prompt = RichPrompt(
        [Text.from_text("start "), Text.from_text("middle"), Text.from_text(" end")]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_rich_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == RichPrompt([Text.from_text("start middle end")])


def test_to_prompt_works_with_tokens() -> None:
    user_prompt = RichPrompt(
        [
            Tokens.from_token_ids([1, 2, 3]),
            Text.from_text("cool"),
            Tokens.from_token_ids([4, 5, 6]),
        ]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_rich_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_with_empty_template() -> None:
    assert PromptTemplate("").to_rich_prompt() == RichPrompt([])


def test_to_prompt_resets_template(prompt_image: Image) -> None:
    template = PromptTemplate("{{image}}")
    placeholder = template.placeholder(prompt_image)
    prompt = template.to_rich_prompt(image=placeholder)

    prompt_with_reset_template = template.to_rich_prompt(image=placeholder)

    assert prompt_with_reset_template != prompt


def test_to_rich_prompt_returns_ranges(prompt_image: Image) -> None:
    embedded_text = "Embedded"
    prefix_items: list[PromptItem] = [
        Text.from_text("Prefix Text Item"),
        prompt_image,
    ]
    prefix_text = "Prefix text"
    prefix_merged = Text.from_text("Merged Prefix Item")
    embedded_merged = Text.from_text("Merged Embedded Item")
    embedded_items: list[PromptItem] = [prompt_image]
    template = PromptTemplate(
        "{{prefix_items}}{{prefix_text}}{% promptrange r1 %}{{embedded_text}}{{embedded_items}}{% endpromptrange %}",
    )

    prompt_data = template.to_rich_prompt(
        prefix_items=template.embed_prompt(RichPrompt([*prefix_items, prefix_merged])),
        prefix_text=prefix_text,
        embedded_text=embedded_text,
        embedded_items=template.embed_prompt(
            RichPrompt([embedded_merged, *embedded_items])
        ),
    )

    ranges = prompt_data.ranges.get("r1")

    assert ranges == [
        PromptRange(
            start=TextCursor(
                item=len(prefix_items),
                position=len(prefix_merged.text) + len(prefix_text),
            ),
            end=PromptItemCursor(item=len(prefix_items) + len(embedded_items)),
        ),
    ]


def test_to_rich_prompt_returns_ranges_for_image_only_prompt(
    prompt_image: Image,
) -> None:
    template = PromptTemplate(
        dedent(
            """
        {%- promptrange r1 -%}
          {%- promptrange r2 -%}
            {{image}}
          {%- endpromptrange -%}
        {%- endpromptrange -%}"""
        ).lstrip()
    )

    prompt_data = template.to_rich_prompt(image=template.placeholder(prompt_image))
    r1 = prompt_data.ranges.get("r1")

    assert r1 == [PromptRange(start=PromptItemCursor(0), end=PromptItemCursor(0))]
    assert r1 == prompt_data.ranges.get("r2")


def test_to_rich_prompt_returns_no_range_with_empty_template() -> None:
    template = PromptTemplate("{% promptrange r1 %}{% endpromptrange %}")

    assert template.to_rich_prompt().ranges.get("r1") == []


def test_to_rich_prompt_returns_no_range_with_range_with_other_ranges() -> None:
    template = PromptTemplate("""text{% promptrange r1 %}{% endpromptrange %}""")
    assert template.to_rich_prompt().ranges.get("r1") == []


def test_to_rich_prompt_returns_no_empty_ranges(prompt_image: Image) -> None:
    template = PromptTemplate(
        "{{image}}{% promptrange r1 %}{% endpromptrange %} Some Text"
    )

    assert (
        template.to_rich_prompt(image=template.placeholder(prompt_image)).ranges.get(
            "r1"
        )
        == []
    )


def test_prompt_template_raises_liquid_error_on_illeagal_range() -> None:
    with raises(LiquidSyntaxError):
        PromptTemplate(
            "{% for i in (1..4) %}{% promptrange r1 %}{% endfor %}{% endpromptrange %}"
        )


def test_to_rich_prompt_returns_multiple_text_ranges_in_for_loop() -> None:
    embedded = "Some Content"
    template = PromptTemplate(
        "{% for i in (1..4) %}{% promptrange r1 %}{{embedded}}{% endpromptrange %}{% endfor %}"
    )

    prompt_data = template.to_rich_prompt(embedded=embedded)

    assert prompt_data.ranges.get("r1") == [
        PromptRange(
            start=TextCursor(item=0, position=i * len(embedded)),
            end=TextCursor(item=0, position=(i + 1) * len(embedded)),
        )
        for i in range(4)
    ]


def test_to_rich_prompt_returns_multiple_image_ranges_in_for_loop(
    prompt_image: Image,
) -> None:
    template = PromptTemplate(
        "{% for i in (1..4) %}{% promptrange r1 %}{{embedded}}{% endpromptrange %}{% endfor %}"
    )

    prompt_data = template.to_rich_prompt(embedded=template.placeholder(prompt_image))

    assert prompt_data.ranges.get("r1") == [
        PromptRange(
            start=PromptItemCursor(item=i),
            end=PromptItemCursor(item=i),
        )
        for i in range(4)
    ]


def test_from_prompt_creates_valid_rich_prompt() -> None:
    prompt = Prompt(items=[Text.from_text("FooBar")])
    ranges = {
        "foo": [
            PromptRange(
                start=TextCursor(item=0, position=0),
                end=TextCursor(item=0, position=5),
            )
        ]
    }

    rich_prompt = RichPrompt.from_prompt(prompt=prompt, ranges=ranges)

    assert isinstance(rich_prompt, RichPrompt)
    assert rich_prompt.items[0] == prompt.items[0]
    assert rich_prompt.ranges == ranges
