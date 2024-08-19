import pytest

from intelligence_layer.core import (
    Language,
    LuminousControlModel,
    NoOpTracer,
    TextChunk,
    TextHighlight,
)
from intelligence_layer.core.detect_language import LanguageNotSupportedError
from intelligence_layer.examples.qa.single_chunk_qa import (
    QaSetup,
    SingleChunkQa,
    SingleChunkQaInput,
)


def test_qa_with_answer(single_chunk_qa: SingleChunkQa) -> None:
    input_text = "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
    input = SingleChunkQaInput(
        chunk=TextChunk(input_text),
        question="What is the name of Paul Nicolas' brother?",
        language=Language("en"),
    )
    output = single_chunk_qa.run(input, NoOpTracer())

    assert output.answer
    assert "Henri" in output.answer
    assert any(
        "Henri" in input_text[highlight.start : highlight.end] and highlight.score == 1
        for highlight in output.highlights
    )


def test_qa_with_no_answer(single_chunk_qa: SingleChunkQa) -> None:
    input = SingleChunkQaInput(
        chunk=TextChunk(
            "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
        ),
        question="What is the capital of Germany?",
    )
    output = single_chunk_qa.run(input, NoOpTracer())

    assert output.answer is None


def test_language_not_supported_exception(single_chunk_qa: SingleChunkQa) -> None:
    input = SingleChunkQaInput(
        chunk=TextChunk(
            "Paul Nicolas stracił matkę w wieku 3 lat, a następnie ojca w 1914 r.[3] Wychowywała go teściowa wraz z bratem Henrim. Karierę piłkarską rozpoczął w klubie Saint-Mandé w 1916 roku. Początkowo grał jako obrońca, ale szybko zdał sobie sprawę, że jego przeznaczeniem jest gra w pierwszym składzie, ponieważ strzelał wiele bramek[3]. Oprócz instynktu bramkarskiego, Nicolas wyróżniał się również silnym charakterem na boisku, a te dwie cechy w połączeniu ostatecznie zwróciły uwagę pana Forta, ówczesnego prezesa klubu Gallia, który podpisał z nim kontrakt jako środkowym napastnikiem w 1916 roku."
        ),
        question="Jaka jest stolica Niemiec?",
        language=Language("pl"),
    )

    with pytest.raises(LanguageNotSupportedError):
        single_chunk_qa.run(input, NoOpTracer())


def test_qa_with_logit_bias_for_no_answer(
    luminous_control_model: LuminousControlModel,
) -> None:
    first_token = "no"
    max_tokens = 5
    config = {
        Language("en"): QaSetup(
            unformatted_instruction='{{question}}\nIf there\'s no answer, say "{{no_answer_text}}". Only answer the question based on the text.',
            no_answer_str=f"{first_token} answer in text",
            no_answer_logit_bias=1000.0,
        )
    }
    single_chunk_qa = SingleChunkQa(
        luminous_control_model, instruction_config=config, maximum_tokens=max_tokens
    )

    input = SingleChunkQaInput(
        chunk=TextChunk(
            "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916."
        ),
        question="When did he lose his mother?",
    )
    output = single_chunk_qa.run(input, NoOpTracer())
    answer = output.answer

    assert answer
    assert "no" in answer.split()[0]


def test_qa_highlights_will_not_become_out_of_bounds(
    single_chunk_qa: SingleChunkQa,
) -> None:
    input_text = """Zubereitung
Ein Hotdog besteht aus einem erwärmten Brühwürstchen in einem länglichen, meist weichen Weizenbrötchen, das üblicherweise getoastet oder gedämpft wird. Das Hotdogbrötchen wird zur Hälfte der Länge nach aufgeschnitten und ggf. erhitzt. Danach legt man das heiße Würstchen hinein und garniert es mit den Saucen (Ketchup, Senf, Mayonnaise usw.). Häufig werden auch noch weitere Zugaben, etwa Röstzwiebeln, Essiggurken, Sauerkraut oder Krautsalat in das Brötchen gegeben.

Varianten

In Dänemark und teilweise auch in Schweden wird der Hotdog mit leuchtend rot eingefärbten Würstchen (Røde Pølser) hergestellt und kogt (gebrüht) oder risted (gebraten) angeboten. Der dänische Hotdog wird meist mit Röstzwiebeln, gehackten Zwiebeln und süßsauer eingelegten Salatgurken-Scheiben und regional mit Rotkohl garniert. Als Saucen werden Ketchup, milder Senf und dänische Remoulade, die Blumenkohl enthält, verwendet. Der bekannteste Imbiss Dänemarks, der Hotdogs verkauft, ist Annies Kiosk.

Weltweit bekannt sind die Hotdog-Stände der schwedischen Möbelhauskette IKEA, an denen im Möbelhaus hinter den Kassen Hot Dogs der schwedischen Variante zum Selberbelegen mit Röstzwiebeln, Gurken und verschiedenen Soßen verkauft werden. Der Hotdogstand in der Filiale gilt weltweit als eine Art Markenzeichen von IKEA. In Deutschland wird das Gericht meist mit Frankfurter oder Wiener Würstchen zubereitet.

In den USA wird der Hotdog meist auf einem Roller Grill gegart. So bekommt die Wurst einen besonderen Grillgeschmack. Amerikanische Hotdogs werden mit speziellen Pickled Gherkins (Gurkenscheiben) und Relishes (Sweet Relish, Hot Pepper Relish oder Corn Relish), häufig mit mildem Senf (Yellow Mustard, die populärste Hotdog-Zutat) oder mit Ketchup serviert. Auch eine Garnitur aus warmem Sauerkraut ist möglich (Nathan’s Famous in New York)."""
    model = LuminousControlModel("luminous-supreme-control")
    qa_task = SingleChunkQa(
        text_highlight=TextHighlight(model=model, granularity=None, clamp=True),
        model=model,
    )
    input = SingleChunkQaInput(
        chunk=TextChunk(input_text),
        question="What is a hot dog",
        language=Language("de"),
    )
    output = qa_task.run(input, NoOpTracer())

    for highlight in output.highlights:
        assert highlight.start >= 0
        assert 0 < highlight.end <= len(input_text)


def test_qa_single_chunk_does_not_crash_when_input_is_empty(
    single_chunk_qa: SingleChunkQa,
) -> None:
    input_text = ""
    input = SingleChunkQaInput(
        chunk=TextChunk(input_text),
        question="something",
        language=Language("de"),
    )
    res = single_chunk_qa.run(input, NoOpTracer())
    assert res.answer is None
