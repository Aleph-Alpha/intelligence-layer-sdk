import pytest
from aleph_alpha_client import (
    Explanation,
    ExplanationResponse,
    Image,
    TargetPromptItemExplanation,
    Text,
    TextPromptItemExplanation,
    TextScore,
)
from pytest import fixture, raises

from intelligence_layer.core import (
    AlephAlphaModel,
    ControlModel,
    ExplainInput,
    ExplainOutput,
    LuminousControlModel,
    NoOpTracer,
    PromptTemplate,
    RichPrompt,
    ScoredTextHighlight,
    TextHighlight,
    TextHighlightInput,
    Tracer,
)


@fixture
def text_highlight(
    luminous_control_model: ControlModel,
) -> TextHighlight:
    return TextHighlight(luminous_control_model)


@fixture
def text_highlight_with_clamp(
    luminous_control_model: ControlModel,
) -> TextHighlight:
    return TextHighlight(luminous_control_model, clamp=True)


# To test with multimodal input, we also need a base model.
@fixture
def text_highlight_base() -> TextHighlight:
    return TextHighlight(AlephAlphaModel("luminous-base"))


def map_to_prompt(prompt: RichPrompt, highlight: ScoredTextHighlight) -> str:
    # This is only used for this test file
    assert isinstance(prompt.items[0], Text)
    return prompt.items[0].text[highlight.start : highlight.end]


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
    assert all(score >= 0 for score in scores)
    max_score, max_index = max((v, i) for i, v in enumerate(scores))
    top_highlight = output.highlights[max_index]
    assert max_score == 1
    assert "conditions" in map_to_prompt(rich_prompt, top_highlight)


def test_text_highlight_indices_start_at_prompt_range(
    text_highlight: TextHighlight,
) -> None:
    answer = " Berlin"
    prompt_template_str = """Question: Name a capital in the text.
Text: {% promptrange r1 %}Berlin{% endpromptrange %}
Answer:"""
    rich_prompt = PromptTemplate(prompt_template_str).to_rich_prompt()

    input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset({"r1"}),
    )
    output = text_highlight.run(input, NoOpTracer())

    scores = [highlight.score for highlight in output.highlights]
    max_index = max((v, i) for i, v in enumerate(scores))[1]
    top_highlight = output.highlights[max_index]
    assert map_to_prompt(rich_prompt, top_highlight) == "Berlin"


def test_text_highlight_with_range_without_highlight(
    text_highlight: TextHighlight,
) -> None:
    answer = " Extreme conditions."
    prompt_template_str = """Question: What is the ecosystem adapted to?
{% promptrange no_content %}Mozart was a musician born on 27th january of 1756. He lived in Wien, Österreich. {% endpromptrange %}
{% promptrange content %}Scientists at the European Southern Observatory announced a groundbreaking discovery today: microbial life detected in the water-rich atmosphere of Proxima Centauri b, our closest neighboring exoplanet.
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
    target_sentence = "The evidence, drawn from a unique spectral signature of organic compounds, hints at an ecosystem adapted to extreme conditions."
    assert not any(
        target_sentence in map_to_prompt(rich_prompt, highlight)
        for highlight in output.highlights
    )
    # highlights should have a low score as they are not relevant to the answer.
    assert len(output.highlights) == 0


def test_text_highlight_with_image_prompt(
    text_highlight_base: TextHighlight, prompt_image: Image
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
    with pytest.raises(ValueError):
        text_highlight_base.run(input, NoOpTracer())


def test_text_highlight_without_range(text_highlight: TextHighlight) -> None:
    prompt_template_str = """Question: What is the Latin name of the brown bear?
Text: The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.
Answer:"""
    template = PromptTemplate(prompt_template_str)
    rich_prompt = template.to_rich_prompt()
    completion = " Ursus arctos."

    input = TextHighlightInput(rich_prompt=rich_prompt, target=completion)
    output = text_highlight.run(input, NoOpTracer())

    assert output.highlights
    assert any(
        "bear" in map_to_prompt(rich_prompt, highlight).lower()
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
    rich_prompt = template.to_rich_prompt()
    answer = " Extreme conditions."

    unknown_range_name = "bla"
    input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset([unknown_range_name]),
    )
    with raises(ValueError) as e:
        text_highlight.run(input, NoOpTracer())

    assert unknown_range_name in str(e.value)


def test_text_ranges_do_not_overlap_into_question_when_clamping(
    text_highlight_with_clamp: TextHighlight,
) -> None:
    instruct = """Beantworte die Frage anhand des Textes. Wenn sich die Frage nicht mit dem Text beantworten lässt, antworte "Unbeantwortbar".\nFrage: What is a hot dog\n"""
    prompt_template_str = """{% promptrange instruction %}{{instruct}}{% endpromptrange %}
{% promptrange input %}Zubereitung \nEin Hotdog besteht aus einem erwärmten Brühwürstchen in einem länglichen, meist weichen Weizenbrötchen, das üblicherweise getoastet oder gedämpft wird. Das Hotdogbrötchen wird zur Hälfte der Länge nach aufgeschnitten und ggf. erhitzt. Danach legt man das heiße Würstchen hinein und garniert es mit den Saucen (Ketchup, Senf, Mayonnaise usw.). Häufig werden auch noch weitere Zugaben, etwa Röstzwiebeln, Essiggurken, Sauerkraut oder Krautsalat in das Brötchen gegeben.\n\nVarianten \n\n In Dänemark und teilweise auch in Schweden wird der Hotdog mit leuchtend rot eingefärbten Würstchen (Røde Pølser) hergestellt und kogt (gebrüht) oder risted (gebraten) angeboten. Der dänische Hotdog wird meist mit Röstzwiebeln, gehackten Zwiebeln und süßsauer eingelegten Salatgurken-Scheiben und regional mit Rotkohl garniert. Als Saucen werden Ketchup, milder Senf und dänische Remoulade, die Blumenkohl enthält, verwendet. Der bekannteste Imbiss Dänemarks, der Hotdogs verkauft, ist Annies Kiosk. \n\nWeltweit bekannt sind die Hotdog-Stände der schwedischen Möbelhauskette IKEA, an denen im Möbelhaus hinter den Kassen Hot Dogs der schwedischen Variante zum Selberbelegen mit Röstzwiebeln, Gurken und verschiedenen Soßen verkauft werden. Der Hotdogstand in der Filiale gilt weltweit als eine Art Markenzeichen von IKEA. In Deutschland wird das Gericht meist mit Frankfurter oder Wiener Würstchen zubereitet.\n\n In den USA wird der Hotdog meist auf einem Roller Grill gegart. So bekommt die Wurst einen besonderen Grillgeschmack. Amerikanische Hotdogs werden mit speziellen Pickled Gherkins (Gurkenscheiben) und Relishes (Sweet Relish, Hot Pepper Relish oder Corn Relish), häufig mit mildem Senf (Yellow Mustard, die populärste Hotdog-Zutat) oder mit Ketchup serviert. Auch eine Garnitur aus warmem Sauerkraut ist möglich (Nathan’s Famous in New York).\n{% endpromptrange %}
### Response:"""
    template = PromptTemplate(prompt_template_str)
    rich_prompt = template.to_rich_prompt(instruct=instruct)
    answer = "Ein Hotdog ist ein Würstchen, das in einem Brötchen serviert wird."

    input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset(["input"]),
    )
    result = text_highlight_with_clamp.run(input, NoOpTracer())
    for highlight in result.highlights:
        assert highlight.start >= len(instruct)
        assert highlight.end > 0


def test_highlight_does_not_clamp_when_prompt_ranges_overlap(
    text_highlight_with_clamp: TextHighlight,
) -> None:
    # given
    prompt_template_str = """{% promptrange outer %}t{% promptrange inner %}es{% endpromptrange %}t{% endpromptrange %}"""
    template = PromptTemplate(prompt_template_str)
    rich_prompt = template.to_rich_prompt()
    answer = "Test?"

    highlight_input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset(["outer", "inner"]),
    )
    # when
    result = text_highlight_with_clamp.run(highlight_input, NoOpTracer())

    # then
    assert result.highlights[0].start == 0
    assert result.highlights[0].end == 4


def test_highlight_clamps_to_the_correct_range_at(
    text_highlight_with_clamp: TextHighlight,
) -> None:
    # given
    prompt_template_str = """{% promptrange short %}t{% endpromptrange %}e{% promptrange long %}st{% endpromptrange %}"""
    template = PromptTemplate(prompt_template_str)
    rich_prompt = template.to_rich_prompt()
    answer = "Test?"

    highlight_input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset(["short", "long"]),
    )
    # when
    result = text_highlight_with_clamp.run(highlight_input, NoOpTracer())

    # then
    assert result.highlights[0].start == 2
    assert result.highlights[0].end == 4


class FakeHighlightModel(LuminousControlModel):
    def __init__(self, model: AlephAlphaModel) -> None:
        self.model = model

    def explain(self, input: ExplainInput, tracer: Tracer) -> ExplainOutput:
        items = [
            TextPromptItemExplanation(scores=[TextScore(start=0, length=4, score=0.5)]),
            TargetPromptItemExplanation(scores=[]),
        ]
        explanations = [Explanation(target=input.target, items=items)]  # type: ignore
        return ExplainOutput.from_explanation_response(
            ExplanationResponse(model_version="dummy", explanations=explanations)
        )


@fixture
def fake_highlight_model(luminous_control_model: ControlModel) -> ControlModel:
    return FakeHighlightModel(luminous_control_model)


@pytest.mark.parametrize(
    "prompt, start, end, score",
    [  # scores taken from test
        ("""t{% promptrange short %}e{% endpromptrange%}st""", 1, 2, 0.5 * 1 / 4),
        ("""{% promptrange short %}te{% endpromptrange%}st""", 0, 2, 0.5 * 2 / 4),
        ("""te{% promptrange short %}st{% endpromptrange%}""", 2, 4, 0.5 * 2 / 4),
        (
            """t{% promptrange short %}est .....{% endpromptrange%}""",
            1,
            4,
            0.5 * 3 / 4,
        ),
    ],
)
def test_highlight_clamps_end_correctly(
    fake_highlight_model: ControlModel,
    prompt: str,
    start: int,
    end: int,
    score: int,
) -> None:
    # given
    text_highlight_with_clamp = TextHighlight(model=fake_highlight_model, clamp=True)
    template = PromptTemplate(prompt)
    rich_prompt = template.to_rich_prompt()
    answer = "Test?"

    highlight_input = TextHighlightInput(
        rich_prompt=rich_prompt,
        target=answer,
        focus_ranges=frozenset(["short"]),
    )
    # when
    result = text_highlight_with_clamp.run(highlight_input, NoOpTracer())

    # then
    assert result.highlights[0].start == start
    assert result.highlights[0].end == end
    assert abs(result.highlights[0].score - score) <= 0.02
