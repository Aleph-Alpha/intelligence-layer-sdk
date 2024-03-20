from dataclasses import dataclass
from threading import Lock
from typing import Mapping, Sequence

import nltk  # type: ignore
from langdetect import detect_langs  # type: ignore
from langdetect.language import Language as LangdetectLanguage  # type: ignore
from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer  # type: ignore
from nltk.translate.bleu_score import sentence_bleu  # type: ignore
from rouge import Rouge  # type: ignore

_nltk_lock = Lock()


def _download_nltk() -> None:
    with _nltk_lock:
        nltk.download("punkt", quiet=True)


def _split_into_words(input: str) -> Sequence[str]:
    """Splits a string into a list of words.

    Removes non-alphanumeric characters and lowercases the given text.

    Args:
        input: String to split.
    Returns:
        List of words.
    """
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(input.lower())
    assert isinstance(tokens, list)
    return tokens


class BleuGrader:
    def __init__(self) -> None:
        _download_nltk()

    def calculate_bleu(self, hypothesis: str, reference: str) -> float:
        """Calculates the BLEU-score for the given hypothesis and reference.

        In the summarization use-case the `BLEU-score <https://aclanthology.org/P02-1040/>`_ roughly corresponds to the precision of the generated summary with regard to the expected summary.

        Args:
            hypothesis: The generation to be evaluated.
            reference: The baseline for the evaluation.

        Returns:
            BLEU-score, float between 0 and 1. Where 1 means perfect match and 0 no overlap.
        """
        hypothesis_tokens = _split_into_words(hypothesis)
        reference_tokens = _split_into_words(reference)
        bleu_score = sentence_bleu(
            references=[reference_tokens], hypothesis=hypothesis_tokens
        )
        return bleu_score if isinstance(bleu_score, float) else 0.0


@dataclass
class RougeScores:
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_rouge_results(cls, rouge_results: Mapping[str, float]) -> "RougeScores":
        return cls(
            precision=rouge_results["p"],
            recall=rouge_results["r"],
            f1=rouge_results["f"],
        )


class RougeGrader:
    def __init__(self) -> None:
        _download_nltk()

    def calculate_rouge(self, hypothesis: str, reference: str) -> RougeScores:
        """Calculates the ROUGE-score for the hypothesis and reference.

        In the summarization use-case the `ROUGE-score <https://aclanthology.org/W04-1013>`_ roughly corresponds to the recall of the generated summary with regard to the expected summary.

        Args:
            hypothesis: The generation to be evaluated.
            reference: The baseline for the evaluation.

        Returns:
            ROUGE-score, which contains precision, recall and f1 metrics, all will be floats between 0 and 1. Where 1 means perfect match and 0 no overlap.
        """
        hypothesis = " ".join(_split_into_words(hypothesis))
        reference = " ".join(_split_into_words(reference))
        rouge = Rouge()
        rouge_scores = rouge.get_scores(hypothesis, reference)[0]["rouge-2"]
        return RougeScores.from_rouge_results(rouge_scores)


class LanguageMatchesGrader:
    """Provides a method to evaluate whether two texts are of the same language

    Args:
        acceptance_threshold: probability a language must surpass to be accepted
    """

    _acceptance_threshold: float

    def __init__(self, acceptance_threshold: float = 0.75) -> None:
        self._acceptance_threshold = acceptance_threshold
        _download_nltk()

    def languages_match(self, input: str, output: str) -> bool:
        """Calculates if the input and output text are of the same language

        Args:
            input: text for which languages is compared to
            output: text

        Returns:
            bool: whether input and output language match
                  returns true if clear input language is not determinable
        """

        dominant_input_language = self._get_dominant_language(input)

        if dominant_input_language is None:
            return True

        dominant_output_language = self._get_dominant_language(output)

        return dominant_input_language == dominant_output_language

    def _get_dominant_language(self, text: str) -> str | None:
        sentences: Sequence[str] = sent_tokenize(text)
        probs_per_language = self._get_scores_per_language(sentences)
        dominant_language = next(
            (
                langs
                for langs, probs in probs_per_language.items()
                if probs >= self._acceptance_threshold
            ),
            None,
        )
        return dominant_language

    @classmethod
    def _get_scores_per_language(cls, sentences: Sequence[str]) -> dict[str, float]:
        scores_per_language: dict[str, float] = {}
        for sentence in sentences:
            languages_with_probs: Sequence[LangdetectLanguage] = detect_langs(sentence)
            for language in languages_with_probs:
                scores_per_language[language.lang] = scores_per_language.get(
                    language.lang, 0
                ) + language.prob * len(sentence)

        return cls._normalize_dict(scores_per_language)

    @staticmethod
    def _normalize_dict(dictionary: dict[str, float]) -> dict[str, float]:
        total = sum(dictionary.values())
        if total == 0:
            return {key: 0 for key in dictionary}
        return {key: value / total for key, value in dictionary.items()}


class HighlightCoverageGrader:
    def compare_highlights(
        self,
        text: str,
        generated_highlights: frozenset[str],
        expected_highlights: frozenset[str]
    ):     
        def get_start_and_stop_index(highlights: frozenset[str]) -> Sequence[tuple[int, int]]:
            return [
                (
                    text.index(highlight),
                    text.index(highlight)+len(highlight)
                ) for highlight in highlights
            ]

        start_and_stop_indices_generated = get_start_and_stop_index(generated_highlights)
        start_and_stop_indices_expected = get_start_and_stop_index(expected_highlights)

        def highlight_ranges(generated_highlights, expected_highlights):
            # Convert set of ranges to list of booleans for each character in text
            def get_highlight_map(highlights):
                highlight_map = [False] * len(text)
                for start, end in highlights:
                    for i in range(start, end):
                        highlight_map[i] = True
                return highlight_map

            gen_map = get_highlight_map(generated_highlights)
            exp_map = get_highlight_map(expected_highlights)

            overlapping, non_overlapping_gen, non_overlapping_exp = [], [], []
            current_range = None

            for i, (g, e) in enumerate(zip(gen_map, exp_map)):
                if g and e:
                    if current_range is None:
                        current_range = [i, i + 1]
                    else:
                        current_range[1] = i + 1
                else:
                    if current_range:
                        overlapping.append(tuple(current_range))
                        current_range = None

                    if g != e:
                        if g:
                            target_list = non_overlapping_gen
                        else:
                            target_list = non_overlapping_exp

                        if target_list and target_list[-1][1] == i:
                            target_list[-1] = (target_list[-1][0], i + 1)
                        else:
                            target_list.append((i, i + 1))

            # Append the last found range if any
            if current_range:
                overlapping.append(tuple(current_range))

            return overlapping, non_overlapping_gen, non_overlapping_exp

        highlight_ranges(
            start_and_stop_indices_generated,
            start_and_stop_indices_expected
        )

