from intelligence_layer.core.detect_language import DetectLanguage, DetectLanguageInput
from intelligence_layer.core.logger import NoOpDebugLogger


def test_detect_language_returns_correct_language():
    text = "Hello, my name is Niklas. I am working with Pit on this language detection piece."
    task = DetectLanguage(threshold=0.5)
    input = DetectLanguageInput(
        text=text, possible_languages=["en", "de", "fr", "it", "es"]
    )
    debug_log = NoOpDebugLogger()
    output = task.run(input, debug_log)

    assert output.best_fit == "en"
    assert len(output.probabilities) == 5
