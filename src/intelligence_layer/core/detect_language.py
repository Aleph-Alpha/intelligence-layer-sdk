from typing import NewType, Optional, Sequence

from langdetect import detect_langs  # type: ignore
from pydantic import BaseModel

from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.task import Task


class LanguageNotSupportedError(RuntimeError):
    """Raised in case language in the input is not compatible with the languages supported in the task"""


Language = NewType("Language", str)
"""A language identified by its `ISO 639-1 code <https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes>`_."""


class DetectLanguageInput(BaseModel):
    """The input for a `DetectLanguage` task.

    Attributes:
        text: The text to identify the language for.
        possible_languages: All languages that should be considered during detection.
            Languages should be provided with their ISO 639-1 codes.
    """

    text: str
    possible_languages: Sequence[Language] = [
        Language("en"),
        Language("de"),
        Language("es"),
        Language("fr"),
        Language("it"),
    ]


class DetectLanguageOutput(BaseModel):
    """The output of a `DetectLanguage` task.

    Attributes:
        best_fit: The prediction for the best matching language.
            Will be `None` if no language has a probability above the threshold.
    """

    best_fit: Optional[Language]


class AnnotatedLanguage(BaseModel):
    lang: Language
    prob: float


class DetectLanguage(Task[DetectLanguageInput, DetectLanguageOutput]):
    """Task that detects the language of a text.

    Analyzes the likelihood that a given text is written in one of the
    `possible_languages`. Returns the best match or `None`.

    Args:
        threshold: Minimum probability value for a language to be considered
            the `best_fit`.

    Example:
        >>> task = DetectLanguage()
        >>> input = DetectLanguageInput(
                text="This is an English text.",
                allowed_langs=[Language(l) for l in ("en", "fr)],
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
        annotated_languages = self._detect_languages(input, logger)
        best_fit = self._get_best_fit(annotated_languages, input.possible_languages)
        return DetectLanguageOutput(best_fit=best_fit)

    def _detect_languages(
        self, input: DetectLanguageInput, logger: DebugLogger
    ) -> Sequence[AnnotatedLanguage]:
        languages = detect_langs(input.text)
        annotated_languages = [
            AnnotatedLanguage(lang=Language(lang.lang), prob=lang.prob)
            for lang in languages
        ]
        logger.log("Raw language probabilities", annotated_languages)
        return annotated_languages

    def _get_best_fit(
        self,
        languages_result: Sequence[AnnotatedLanguage],
        possible_languages: Sequence[Language],
    ) -> Optional[Language]:
        return (
            languages_result[0].lang
            if (
                languages_result[0].prob >= self._threshold
                and languages_result[0].lang in possible_languages
            )
            else None
        )
