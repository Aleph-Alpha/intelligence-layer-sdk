from intelligence_layer.core.detect_language import (
    DetectLanguage,
    DetectLanguageInput,
    Language,
)
from intelligence_layer.core.tracer import NoOpTracer


def test_detect_language_returns_correct_language() -> None:
    text = "Hello, my name is Niklas. I am working with Pit on this language detection piece."
    task = DetectLanguage()
    input = DetectLanguageInput(
        text=text,
        possible_languages=[Language(lang) for lang in ["en", "de", "fr", "it", "es"]],
    )
    tracer = NoOpTracer()
    output = task.run(input, tracer)

    assert output.best_fit == "en"
