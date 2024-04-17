import pytest

from intelligence_layer.core import (
    DetectLanguage,
    DetectLanguageInput,
    Language,
    NoOpTracer,
)


@pytest.mark.parametrize(
    "text_input,expected_language",
    [
        (
            "Hello, my name is Niklas. I am working with Pit on this language detection piece.",
            Language("en"),
        ),
        (
            "Hola, mi nombre es Niklas. Estoy trabajando con Pit en esta pieza de detección de idioma.",
            Language("es"),
        ),
        (
            "Ciao, mi chiamo Niklas. Sto lavorando con Pit su questo pezzo di rilevamento della lingua.",
            Language("it"),
        ),
        (
            "Hallo, mein Name ist Niklas. Ich arbeite mit Pit an diesem Stück zur Spracherkennung.",
            Language("de"),
        ),
        (
            "Bonjour, je m'appelle Niklas. Je travaille avec Pit sur cette pièce de détection de langue.",
            Language("fr"),
        ),
    ],
)
def test_detect_language_returns_correct_language(
    text_input: str, expected_language: Language
) -> None:
    task = DetectLanguage()
    input = DetectLanguageInput(
        text=text_input,
        possible_languages=[Language(lang) for lang in ["en", "de", "fr", "it", "es"]],
    )
    tracer = NoOpTracer()
    output = task.run(input, tracer)

    assert output.best_fit == expected_language


def test_detect_language_returns_non_if_no_language_can_be_detected() -> None:
    text = "Je m’appelle Jessica. Je suis une fille, je suis française et j’ai treize ans."  # codespell:ignore
    task = DetectLanguage()
    input = DetectLanguageInput(
        text=text,
        possible_languages=[Language(lang) for lang in ["en", "de"]],
    )
    tracer = NoOpTracer()
    output = task.run(input, tracer)

    assert output.best_fit is None
