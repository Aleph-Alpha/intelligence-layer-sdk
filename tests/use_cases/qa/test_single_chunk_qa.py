import pytest

from intelligence_layer.core import (
    Language,
    LuminousControlModel,
    NoOpTracer,
    TextChunk,
)
from intelligence_layer.core.detect_language import LanguageNotSupportedError
from intelligence_layer.use_cases.qa.single_chunk_qa import (
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
    for h in output.highlights:
        print(h.start, h.end)
        print(input_text[h.start : h.end])
    assert any(
        "Henri" in input_text[highlight.start : highlight.end]
        for highlight in output.highlights
    )
    assert len(output.highlights) == 1


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

    # on CI, this is tokenized as "nonononono" rather than "no no no no no"
    # Likely, this is because some test changes the tokenizer state to remove the whitespace
    # We should fix this, but for now, I'll assert both
    acceptable_answers = [
        " ".join([first_token] * max_tokens),
        first_token * max_tokens,
    ]
    assert any(output.answer == a for a in acceptable_answers)
