from aleph_alpha_client import Image
from pytest import fixture, raises

from intelligence_layer.core.model import AlephAlphaModel
from intelligence_layer.core.prompt_template import PromptTemplate, RichPrompt
from intelligence_layer.core.text_highlight import TextHighlight, TextHighlightInput
from intelligence_layer.core.tracer import NoOpTracer


class AlephAlphaVanillaModel(AlephAlphaModel):
    def to_instruct_prompt(
        self,
        instruction: str,
        input: str | None = None,
        response_prefix: str | None = None,
    ) -> RichPrompt:
        raise NotImplementedError()


@fixture
def aleph_alpha_vanilla_model() -> AlephAlphaVanillaModel:
    return AlephAlphaVanillaModel("luminous-base")


@fixture
def text_highlight(aleph_alpha_vanilla_model: AlephAlphaVanillaModel) -> TextHighlight:
    return TextHighlight(aleph_alpha_vanilla_model)


def test_text_highlight(text_highlight: TextHighlight) -> None:
    answer = " Extreme conditions."
    prompt_template_str = """Question: What is the ecosystem adapted to?
Text: {% promptrange r1 %}Scientists at the European Southern Observatory announced a groundbreaking discovery today: microbial life detected in the water-rich atmosphere of Proxima Centauri b, our closest neighboring exoplanet.
The evidence, drawn from a unique spectral signature of organic compounds, hints at an ecosystem adapted to extreme conditions.
This finding, while not complex extraterrestrial life, significantly raises the prospects of life's commonality in the universe.
The international community is abuzz with plans for more focused research and potential interstellar missions.{% endpromptrange %}
Answer:"""
    prompt_with_metadata = PromptTemplate(prompt_template_str).to_rich_prompt()

    input = TextHighlightInput(
        rich_prompt=prompt_with_metadata,
        target=answer,
        focus_ranges=frozenset({"r1"}),
    )
    output = text_highlight.run(input, NoOpTracer())

    assert output.highlights
    top_highlight = next(h for h in output.highlights if "conditions" in h.text)
    assert all(
        top_highlight.score >= highlight.score for highlight in output.highlights
    )


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
    prompt_with_metadata = PromptTemplate(prompt_template_str).to_rich_prompt(
        answer=answer
    )

    input = TextHighlightInput(
        rich_prompt=prompt_with_metadata,
        target=f" {answer}",
        focus_ranges=frozenset(["no_content"]),
    )
    output = text_highlight.run(input, NoOpTracer())
    assert not any(h.score > 0 for h in output.highlights)


def test_text_highlight_with_only_one_sentence(text_highlight: TextHighlight) -> None:
    prompt_template_str = """What is the Latin name of the brown bear? The answer is Ursus Arctos.{% promptrange r1 %} Explanation should not highlight anything.{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_rich_prompt()
    completion = " Ursus Arctos"

    input = TextHighlightInput(
        rich_prompt=prompt_with_metadata,
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
    prompt_with_metadata = template.to_rich_prompt(
        image=template.placeholder(prompt_image)
    )
    completion = " The latin name of the brown bear is Ursus arctos."

    input = TextHighlightInput(rich_prompt=prompt_with_metadata, target=completion)
    output = text_highlight.run(input, NoOpTracer())

    assert output.highlights
    assert any("bear" in highlight.text.lower() for highlight in output.highlights)


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
    assert any("bear" in highlight.text.lower() for highlight in output.highlights)


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
        "conditions" in highlight.text.lower() for highlight in output.highlights
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
