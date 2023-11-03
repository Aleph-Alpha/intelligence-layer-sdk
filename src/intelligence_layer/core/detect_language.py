from typing import Mapping, Optional, Sequence

from langdetect import detect_langs, language
from pydantic import BaseModel

from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.task import Task


class DetectLanguageInput(BaseModel):
    text: str
    possible_languages: Sequence[str]


class DetectLanguageOutput(BaseModel):
    best_fit: Optional[str]
    probabilities: Mapping[str, float]


class DetectLanguage(Task[DetectLanguageInput, DetectLanguageOutput]):
    def __init__(self, threshold: float):
        super().__init__()
        self._threshold = threshold

    def run(
        self, input: DetectLanguageInput, logger: DebugLogger
    ) -> DetectLanguageOutput:
        languages = detect_langs(input.text)
        best_fit = self._get_best_fit(languages, input.possible_languages)
        probabilities = self._get_probabilities(languages, input.possible_languages)
        return DetectLanguageOutput(best_fit=best_fit, probabilities=probabilities)

    def _get_best_fit(
        self,
        languages_result: list[language.Language],
        possible_languages: Sequence[str],
    ) -> str:
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
        languages_result: list[language.Language],
        possible_languages: Sequence[str],
    ) -> Mapping[str, float]:
        def get_prob(target_lang: str) -> float:
            for l in languages_result:
                if l.lang == target_lang:
                    return l.prob
            return 0.0

        return {l: get_prob(l) for l in possible_languages}
