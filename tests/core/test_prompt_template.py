from pathlib import Path
from textwrap import dedent
from typing import List

from aleph_alpha_client.prompt import Prompt, Image, PromptItem, Text, Tokens
from liquid.exceptions import LiquidTypeError, LiquidSyntaxError
from pytest import raises

from intelligence_layer.core.prompt_template import (
    PromptItemCursor,
    PromptRange,
    PromptTemplate,
    TextCursor,
)


def test_to_prompt_with_text_array() -> None:
    template = PromptTemplate(
        """
{%- for name in names -%}
Hello {{name}}!
{% endfor -%}
        """
    )
    names = ["World", "Rutger"]

    prompt = template.to_prompt(names=names)

    expected = "".join([f"Hello {name}!\n" for name in names])
    assert prompt == Prompt.from_text(expected)


def test_to_prompt_with_invalid_input() -> None:
    template = PromptTemplate(
        """
{%- for name in names -%}
Hello {{name}}!
{% endfor -%}
        """
    )

    with raises(LiquidTypeError):
        template.to_prompt(names=7)


def test_to_prompt_with_single_image(prompt_image: Image) -> None:
    template = PromptTemplate(
        """Some Text.
{{whatever}}
More Text
"""
    )

    prompt = template.to_prompt(whatever=template.placeholder(prompt_image))

    expected = Prompt(
        [
            Text.from_text("Some Text.\n"),
            prompt_image,
            Text.from_text("\nMore Text\n"),
        ]
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

    prompt = template.to_prompt(
        images=[template.placeholder(prompt_image), template.placeholder(prompt_image)]
    )

    expected = Prompt([prompt_image, prompt_image])
    assert prompt == expected


def test_to_prompt_with_mixed_modality_variables(prompt_image: Image) -> None:
    template = PromptTemplate("""{{image}}{{name}}{{image}}""")

    prompt = template.to_prompt(
        image=template.placeholder(prompt_image), name="whatever"
    )

    expected = Prompt([prompt_image, Text.from_text("whatever"), prompt_image])
    assert prompt == expected


def test_to_prompt_with_unused_image(prompt_image: Image) -> None:
    template = PromptTemplate("cool")

    prompt = template.to_prompt(images=template.placeholder(prompt_image))

    assert prompt == Prompt.from_text("cool")


def test_to_prompt_with_multiple_different_images(prompt_image: Image) -> None:
    image_source_path = Path(__file__).parent.parent / "image_example.jpg"
    second_image = Image.from_file(image_source_path)

    template = PromptTemplate("""{{image_1}}{{image_2}}""")

    prompt = template.to_prompt(
        image_1=template.placeholder(prompt_image),
        image_2=template.placeholder(second_image),
    )

    assert prompt == Prompt([prompt_image, second_image])


def test_to_prompt_with_embedded_prompt(prompt_image: Image) -> None:
    user_prompt = Prompt([Text.from_text("Cool"), prompt_image])

    template = PromptTemplate("""{{user_prompt}}""")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_does_not_add_whitespace_after_image(prompt_image: Image) -> None:
    user_prompt = Prompt([prompt_image, Text.from_text("Cool"), prompt_image])

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_skips_empty_strings() -> None:
    user_prompt = Prompt(
        [Text.from_text("Cool"), Text.from_text(""), Text.from_text("Also cool")]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == Prompt([Text.from_text("Cool Also cool")])


def test_to_prompt_adds_whitespaces() -> None:
    user_prompt = Prompt(
        [Text.from_text("start "), Text.from_text("middle"), Text.from_text(" end")]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == Prompt([Text.from_text("start middle end")])


def test_to_prompt_works_with_tokens() -> None:
    user_prompt = Prompt(
        [
            Tokens.from_token_ids([1, 2, 3]),
            Text.from_text("cool"),
            Tokens.from_token_ids([4, 5, 6]),
        ]
    )

    template = PromptTemplate("{{user_prompt}}")

    prompt = template.to_prompt(user_prompt=template.embed_prompt(user_prompt))

    assert prompt == user_prompt


def test_to_prompt_with_empty_template() -> None:
    assert PromptTemplate("").to_prompt() == Prompt([])


def test_to_prompt_resets_template(prompt_image: Image) -> None:
    template = PromptTemplate("{{image}}")
    placeholder = template.placeholder(prompt_image)
    prompt = template.to_prompt(image=placeholder)

    prompt_with_reset_template = template.to_prompt(image=placeholder)

    assert prompt_with_reset_template != prompt


def test_to_prompt_data_returns_ranges(prompt_image: Image) -> None:
    embedded_text = "Embedded"
    prefix_items: List[PromptItem] = [
        Text.from_text("Prefix Text Item"),
        prompt_image,
    ]
    prefix_text = "Prefix text"
    prefix_merged = Text.from_text("Merged Prefix Item")
    embedded_merged = Text.from_text("Merged Embedded Item")
    embedded_items: List[PromptItem] = [prompt_image]
    template = PromptTemplate(
        "{{prefix_items}}{{prefix_text}}{% promptrange r1 %}{{embedded_text}}{{embedded_items}}{% endpromptrange %}",
    )

    prompt_data = template.to_prompt_with_metadata(
        prefix_items=template.embed_prompt(Prompt(prefix_items + [prefix_merged])),
        prefix_text=prefix_text,
        embedded_text=embedded_text,
        embedded_items=template.embed_prompt(
            Prompt([embedded_merged] + embedded_items)
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


def test_to_prompt_data_returns_ranges_for_image_only_prompt(
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

    prompt_data = template.to_prompt_with_metadata(
        image=template.placeholder(prompt_image)
    )
    r1 = prompt_data.ranges.get("r1")

    assert r1 == [PromptRange(start=PromptItemCursor(0), end=PromptItemCursor(0))]
    assert r1 == prompt_data.ranges.get("r2")


def test_to_prompt_data_returns_no_range_with_empty_template() -> None:
    template = PromptTemplate("{% promptrange r1 %}{% endpromptrange %}")

    assert template.to_prompt_with_metadata().ranges.get("r1") == []


def test_to_prompt_data_returns_no_empty_ranges(prompt_image: Image) -> None:
    template = PromptTemplate(
        "{{image}}{% promptrange r1 %}{% endpromptrange %} Some Text"
    )

    assert (
        template.to_prompt_with_metadata(
            image=template.placeholder(prompt_image)
        ).ranges.get("r1")
        == []
    )


def test_prompt_template_raises_liquid_error_on_illeagal_range() -> None:
    with raises(LiquidSyntaxError):
        PromptTemplate(
            "{% for i in (1..4) %}{% promptrange r1 %}{% endfor %}{% endpromptrange %}"
        )


def test_to_prompt_data_returns_multiple_text_ranges_in_for_loop() -> None:
    embedded = "Some Content"
    template = PromptTemplate(
        "{% for i in (1..4) %}{% promptrange r1 %}{{embedded}}{% endpromptrange %}{% endfor %}"
    )

    prompt_data = template.to_prompt_with_metadata(embedded=embedded)

    assert prompt_data.ranges.get("r1") == [
        PromptRange(
            start=TextCursor(item=0, position=i * len(embedded)),
            end=TextCursor(item=0, position=(i + 1) * len(embedded)),
        )
        for i in range(4)
    ]


def test_to_prompt_data_returns_multiple_imgae_ranges_in_for_loop(
    prompt_image: Image,
) -> None:
    template = PromptTemplate(
        "{% for i in (1..4) %}{% promptrange r1 %}{{embedded}}{% endpromptrange %}{% endfor %}"
    )

    prompt_data = template.to_prompt_with_metadata(
        embedded=template.placeholder(prompt_image)
    )

    assert prompt_data.ranges.get("r1") == [
        PromptRange(
            start=PromptItemCursor(item=i),
            end=PromptItemCursor(item=i),
        )
        for i in range(4)
    ]
