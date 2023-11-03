from typing import Mapping, Optional, Sequence

from langdetect import detect_langs  # type: ignore
from pydantic import BaseModel

from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.task import Task


class DetectLanguageInput(BaseModel):
    """The input for a `DetectLanguage` task.

    Attributes:
        text: The text to identify the language for.
        possible_languages: All languages that should be considered during detection.
            Languages should be provided with their ISO 639-1 codes.
    """

    text: str
    possible_languages: Sequence[str]


class DetectLanguageOutput(BaseModel):
    """The output of a `DetectLanguage` task.

    Attributes:
        best_fit: The prediction for the best matching language.
            Will be `None` if no language has a probability above the threshold.
        probabilities: Each possible language with the corresponding probability.
    """

    best_fit: Optional[str]
    probabilities: Mapping[str, float]


class AnnotatedLanguage(BaseModel):
    lang: str
    prob: float


class DetectLanguage(Task[DetectLanguageInput, DetectLanguageOutput]):
    """Task that detects the language of a text.

    Analyzes the likelihood of that a given text is written in one of the
    `possible_languages`.

    Args:
        threshold: Minimum probability value for a language to be considered
            the `best_fit`.

    Example:
        >>> task = DetectLanguage()
        >>> input = DetectLanguageInput(
                text="This is an English text.",
                allowed_langs=["en", "fr],
            )
        >>> logger = InMemoryLogger(name="DetectLanguage")
        >>> output = task.run(input, logger)
        >>> print(output.best_fit)
        en
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self._threshold = threshold

    def run(
        self, input: DetectLanguageInput, logger: DebugLogger
    ) -> DetectLanguageOutput:
        languages = detect_langs(input.text)
        annotated_languages = [
            AnnotatedLanguage(lang=l.lang, prob=l.prob) for l in languages
        ]
        best_fit = self._get_best_fit(annotated_languages, input.possible_languages)
        probabilities = self._get_probabilities(
            annotated_languages, input.possible_languages
        )
        return DetectLanguageOutput(best_fit=best_fit, probabilities=probabilities)

    def _get_best_fit(
        self,
        languages_result: Sequence[AnnotatedLanguage],
        possible_languages: Sequence[str],
    ) -> Optional[str]:
        return (
            languages_result[0].lang
            if (
                languages_result[0].prob >= self._threshold
                and languages_result[0].lang in possible_languages
            )
            else None
        )

    def _get_probabilities(
        self,
        languages_result: Sequence[AnnotatedLanguage],
        possible_languages: Sequence[str],
    ) -> Mapping[str, float]:
        def get_prob(target_lang: str) -> float:
            for l in languages_result:
                if l.lang == target_lang:
                    return l.prob
            return 0.0

        return {l: get_prob(l) for l in possible_languages}
