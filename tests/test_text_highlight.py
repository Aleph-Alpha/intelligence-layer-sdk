from aleph_alpha_client import Client, Image
from pytest import fixture
from intelligence_layer.text_highlight import TextHighlight, TextHighlightInput
from intelligence_layer.prompt_template import PromptTemplate

from intelligence_layer.task import NoOpDebugLogger


@fixture
def text_highlight(client: Client) -> TextHighlight:
    return TextHighlight(client)


def test_text_highlight(text_highlight: TextHighlight) -> None:
    prompt_template_str = """Question: What is the Latin name of the brown bear?{% promptrange r1 %} Explanation should only highlight this. Latin name: Ursus Arctos.{% endpromptrange %} This should also not be highlighted.
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata()
    completion = " Ursus Arctos"
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata, target=completion, model=model
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert output.highlights
    top_highlight = next(h for h in output.highlights if "Ursus" in h.text)
    assert all(
        top_highlight.score >= highlight.score for highlight in output.highlights
    )


def test_text_highlight_with_only_one_sentence(text_highlight: TextHighlight) -> None:
    prompt_template_str = """What is the Latin name of the brown bear?{% promptrange r1 %} Explanation should not highlight anything.{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata()
    completion = " Ursus Arctos"
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata, target=completion, model=model
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert not output.highlights


def test_text_highlight_with_image_prompt(
    text_highlight: TextHighlight, prompt_image: Image
) -> None:
    prompt_template_str = """Question: {% promptrange question %}What is the Latin name of the brown bear?{% endpromptrange %}
Text: {% promptrange text %}The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.
Here is an image, just for LOLs: {{image}}
{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata(
        image=template.placeholder(prompt_image)
    )
    completion = " The latin name of the brown bear is Ursus arctos."
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata, target=completion, model=model
    )
    output = text_highlight.run(input, NoOpDebugLogger())

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
    prompt_with_metadata = template.to_prompt_with_metadata(
        image=template.placeholder(prompt_image)
    )
    completion = " The latin name of the brown bear is Ursus arctos."
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata, target=completion, model=model
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert output.highlights
    assert any("bear" in highlight.text.lower() for highlight in output.highlights)
