from aleph_alpha_client import Image
from pytest import fixture, mark, raises

from intelligence_layer.connectors import AlephAlphaClientProtocol
from intelligence_layer.core import (
    ControlModel,
    NoOpTracer,
    PromptTemplate,
    RichPrompt,
    TextHighlight,
    TextHighlightInput,
)


@fixture
def text_highlight(
    luminous_control_model: ControlModel,
) -> TextHighlight:
    return TextHighlight(luminous_control_model)


def test_text_highlight(text_highlight: TextHighlight) -> None:
    answer = " Extreme conditions."
    prompt_template_str = """Question: What is the ecosystem adapted to?
Text: {% promptrange r1 %}Scientists at the European Southern Observatory announced a groundbreaking discovery today: microbial life detected in the water-rich atmosphere of Proxima Centauri b, our closest neighboring exoplanet.
The evidence, drawn from a unique spectral signature of organic compounds, hints at an ecosystem adapted to extreme conditions.
This finding, while not complex extraterrestrial life, significantly raises the prospects of life's commonality in the universe.
The international community is abuzz with plans for more focused research and potential interstellar missions.{% endpromptrange %}
Answer:"""
    rich_prompt = PromptTemplate(prompt_template_str).to_rich_prompt()

    input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset({"r1"}),
    )
    output = text_highlight.run(input, NoOpTracer())

    assert output.highlights
    scores = [highlight.score for highlight in output.highlights]
    max_score, max_index = max((v, i) for i, v in enumerate(scores))
    top_highlight = output.highlights[max_index]
    assert max_score == 1
    assert "conditions" in prompt_template_str[top_highlight.start : top_highlight.end]


def test_text_highlight_with_range_without_highlight(
    text_highlight: TextHighlight,
) -> None:
    answer = " Extreme conditions."
    prompt_template_str = """Question: What is the ecosystem adapted to?
Text 1: {% promptrange no_content %}This is an unrelated sentence. And here is another one.{% endpromptrange %}
Text 2: {% promptrange content %}Scientists at the European Southern Observatory announced a groundbreaking discovery today: microbial life detected in the water-rich atmosphere of Proxima Centauri b, our closest neighboring exoplanet.
The evidence, drawn from a unique spectral signature of organic compounds, hints at an ecosystem adapted to extreme conditions.
This finding, while not complex extraterrestrial life, significantly raises the prospects of life's commonality in the universe.
The international community is abuzz with plans for more focused research and potential interstellar missions.{% endpromptrange %}
Answer:"""
    rich_prompt = PromptTemplate(prompt_template_str).to_rich_prompt(answer=answer)

    input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset(["no_content"]),
    )
    output = text_highlight.run(input, NoOpTracer())
    assert not any(h.score > 0 for h in output.highlights)


@mark.skip()  # TODO this test does not make any sense to us
def test_text_highlight_with_only_one_sentence(
    text_highlight: TextHighlight,
) -> None:
    prompt_template_str = """What is the Latin name of the brown bear? The answer is Ursus Arctos.{% promptrange r1 %} Explanation should not highlight anything.{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    rich_prompt = template.to_rich_prompt()
    completion = " Ursus Arctos"

    input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=completion,
        focus_ranges=frozenset({"r1"}),
    )
    output = text_highlight.run(input, NoOpTracer())

    assert not output.highlights


def test_text_highlight_with_image_prompt(
    text_highlight: TextHighlight, prompt_image: Image
) -> None:
    prompt_template_str = """Question: {% promptrange question %}What is the Latin name of the brown bear?{% endpromptrange %}
Text: {% promptrange text %}The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.
Here is an image, just for LOLs: {{image}}{range}abc{{image}}{range}
{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    rich_prompt = template.to_rich_prompt(image=template.placeholder(prompt_image))
    completion = " The latin name of the brown bear is Ursus arctos."

    input = TextHighlightInput(rich_prompt=rich_prompt, target=completion)
    output = text_highlight.run(input, NoOpTracer())

    assert output.highlights
    assert any(
        "bear" in prompt_template_str[highlight.start : highlight.end].lower()
        for highlight in output.highlights
    )


def test_text_highlight_without_range(
    text_highlight: TextHighlight, prompt_image: Image
) -> None:
    prompt_template_str = """Question: What is the Latin name of the brown bear?
Text: The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.
Here is an image, just for LOLs: {{image}}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_rich_prompt(
        image=template.placeholder(prompt_image)
    )
    completion = " The latin name of the brown bear is Ursus arctos."

    input = TextHighlightInput(rich_prompt=prompt_with_metadata, target=completion)
    output = text_highlight.run(input, NoOpTracer())

    assert output.highlights
    assert any(
        "bear" in prompt_template_str[highlight.start : highlight.end].lower()
        for highlight in output.highlights
    )


def test_text_highlight_without_focus_range_highlights_entire_prompt(
    text_highlight: TextHighlight, prompt_image: Image
) -> None:
    prompt_template_str = """Question: What is the ecosystem adapted to?
Text: {% promptrange r1 %}Scientists at the European Southern Observatory announced a groundbreaking discovery today: microbial life detected in the water-rich atmosphere of Proxima Centauri b, our closest neighboring exoplanet.
The evidence, drawn from a unique spectral signature of organic compounds, hints at an ecosystem adapted to extreme conditions.
This finding, while not complex extraterrestrial life, significantly raises the prospects of life's commonality in the universe.
The international community is abuzz with plans for more focused research and potential interstellar missions.{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_rich_prompt(
        image=template.placeholder(prompt_image)
    )
    answer = " Extreme conditions."
    focus_ranges: frozenset[str] = frozenset()  # empty

    input = TextHighlightInput(
        rich_prompt=prompt_with_metadata,
        target=answer,
        focus_ranges=focus_ranges,
    )
    output = text_highlight.run(input, NoOpTracer())

    assert output.highlights
    assert any(
        "conditions" in prompt_template_str[highlight.start : highlight.end].lower()
        for highlight in output.highlights
    )


def test_text_highlight_with_unknown_range_raises(
    text_highlight: TextHighlight,
) -> None:
    prompt_template_str = """Question: What is the ecosystem adapted to?
Text: {% promptrange r1 %}Scientists at the European Southern Observatory announced a groundbreaking discovery today: microbial life detected in the water-rich atmosphere of Proxima Centauri b, our closest neighboring exoplanet.
The evidence, drawn from a unique spectral signature of organic compounds, hints at an ecosystem adapted to extreme conditions.
This finding, while not complex extraterrestrial life, significantly raises the prospects of life's commonality in the universe.
The international community is abuzz with plans for more focused research and potential interstellar missions.{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_rich_prompt()
    answer = " Extreme conditions."

    unknown_range_name = "bla"
    input = TextHighlightInput(
        rich_prompt=prompt_with_metadata,
        target=answer,
        focus_ranges=frozenset([unknown_range_name]),
    )
    with raises(ValueError) as e:
        text_highlight.run(input, NoOpTracer())

    assert unknown_range_name in str(e.value)
