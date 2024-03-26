import math
from dataclasses import dataclass
from threading import Lock
from typing import List, Mapping, Sequence, Tuple, cast

import nltk  # type: ignore
from langdetect import LangDetectException, detect_langs  # type: ignore
from langdetect.language import Language as LangdetectLanguage  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from nltk.translate.bleu_score import sentence_bleu  # type: ignore
from rouge import Rouge  # type: ignore
from semantic_text_splitter import TextSplitter

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
class FScores:
    precision: float
    recall: float
    f_score: float

    @classmethod
    def from_rouge_results(cls, rouge_results: Mapping[str, float]) -> "FScores":
        return cls(
            precision=rouge_results["p"],
            recall=rouge_results["r"],
            f_score=rouge_results["f"],
        )


class RougeGrader:
    def __init__(self) -> None:
        _download_nltk()

    def calculate_rouge(self, hypothesis: str, reference: str) -> FScores:
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
        return FScores.from_rouge_results(rouge_scores)


class LanguageMatchesGrader:
    """Provides a method to evaluate whether two texts are of the same language.

    Args:
        acceptance_threshold: probability a language must surpass to be accepted
    """

    _acceptance_threshold: float

    def __init__(self, acceptance_threshold: float = 0.75) -> None:
        self._acceptance_threshold = acceptance_threshold
        _download_nltk()

    def languages_match(self, input: str, output: str) -> bool:
        """Calculates if the input and output text are of the same language.
        The length of the texts and its sentences should be reasonably long in order for good performance.

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
        test_chunks: Sequence[str] = self._tokenize_text(text)
        probs_per_language = self._get_scores_per_language(test_chunks)
        dominant_language = next(
            (
                langs
                for langs, probs in probs_per_language.items()
                if probs >= self._acceptance_threshold
            ),
            None,
        )
        return dominant_language

    @staticmethod
    def _tokenize_text(
        text: str, lower_char_bound: int = 30, upper_char_bound: int = 200
    ) -> Sequence[str]:
        text_splitter = TextSplitter()
        return text_splitter.chunks(text, (lower_char_bound, upper_char_bound))

    @classmethod
    def _get_scores_per_language(cls, text_chunks: Sequence[str]) -> dict[str, float]:
        scores_per_language: dict[str, float] = {}
        for text_chunk in text_chunks:
            try:
                languages_with_probs: Sequence[LangdetectLanguage] = detect_langs(
                    text_chunk
                )
                for language in languages_with_probs:
                    scores_per_language[language.lang] = scores_per_language.get(
                        language.lang, 0
                    ) + language.prob * len(text_chunk)
            except LangDetectException:
                continue  # skip text_chunk in case language cannot be determined
        return cls._normalize_dict(scores_per_language)

    @staticmethod
    def _normalize_dict(dictionary: dict[str, float]) -> dict[str, float]:
        total = sum(dictionary.values())
        if total == 0:
            return {key: 0 for key in dictionary}
        return {key: value / total for key, value in dictionary.items()}


@dataclass
class IndexRange:
    start: int
    stop: int


_HighlightRange = List[IndexRange]


class HighlightCoverageGrader:
    """Evaluates how well the generated highlights match the expected highlights (via precision, recall and f1-score)

    Args:
        beta_factor: factor to control weight of precision (0 <= beta < 1) vs. recall (beta > 1) when computing the f-score
    """

    beta_factor: float

    def __init__(self, beta_factor: float = 1.0) -> None:
        self.beta_factor = beta_factor

    def compute_fscores(
        self,
        generated_highlight_indices: Sequence[Tuple[int, int]],
        expected_highlight_indices: Sequence[Tuple[int, int]],
    ) -> FScores:
        """Calculates how well the generated highlight ranges match the expected ones

        Args:
            generated_highlight_indices: list of tuples(start, end) of the generated highlights
            expected_highlight_indices: list of tuples(start, end) of the generated highlights

        Returns:
            FScores, which contains precision, recall and f-score metrics, all will be floats between 0 and 1,
            where 1 means perfect match and 0 no overlap
        """

        generated_highlight_ranges: _HighlightRange = [
            IndexRange(el[0], el[1]) for el in generated_highlight_indices
        ]
        expected_highlight_ranges: _HighlightRange = [
            IndexRange(el[0], el[1]) for el in expected_highlight_indices
        ]

        (
            correctly_identified_indices,
            false_positive_indices,
            false_negative_indices,
        ) = self._identify_overlap_ranges(
            generated_highlight_ranges, expected_highlight_ranges
        )

        true_positive_length = sum(
            [
                index_range.stop - index_range.start
                for index_range in correctly_identified_indices
            ]
        )
        false_positive_length = sum(
            [
                index_range.stop - index_range.start
                for index_range in false_positive_indices
            ]
        )
        false_negative_length = sum(
            [
                index_range.stop - index_range.start
                for index_range in false_negative_indices
            ]
        )

        precision = true_positive_length / (
            true_positive_length + false_positive_length
        )
        recall = true_positive_length / (true_positive_length + false_negative_length)

        denominator = math.pow(self.beta_factor, 2) * precision + recall
        if denominator == 0.0:
            f1_score = 0.0
        else:
            f1_score = (
                (1 + math.pow(self.beta_factor, 2)) * precision * recall
            ) / denominator

        return FScores(precision=precision, recall=recall, f_score=f1_score)

    @staticmethod
    def _identify_overlap_ranges(
        generated_highlights: _HighlightRange, expected_highlights: _HighlightRange
    ) -> Tuple[_HighlightRange, _HighlightRange, _HighlightRange]:
        max_index: int = max(
            index_range.stop
            for index_range in generated_highlights + expected_highlights
        )

        def get_highlight_present_array(highlights: _HighlightRange) -> Sequence[bool]:
            highlight_map = [False] * max_index
            for index_range in highlights:
                for index in range(index_range.start, index_range.stop):
                    highlight_map[index] = True
            return highlight_map

        gen_highlight_present_array = get_highlight_present_array(generated_highlights)
        exp_highlight_present_array = get_highlight_present_array(expected_highlights)

        overlapping_indices: _HighlightRange = []
        leftover_gen_highlights: _HighlightRange = []
        leftover_exp_highlights: _HighlightRange = []
        current_range: IndexRange | None = None

        for index, (
            generated_highlight_present,
            expected_highlight_present,
        ) in enumerate(zip(gen_highlight_present_array, exp_highlight_present_array)):
            if generated_highlight_present and expected_highlight_present:
                current_range = HighlightCoverageGrader._increase_current_range_by_one(
                    current_range, index
                )
            else:
                if current_range:
                    overlapping_indices.append(current_range)
                    current_range = None

                if generated_highlight_present != expected_highlight_present:
                    if generated_highlight_present:
                        leftover_highlights = leftover_gen_highlights
                    else:
                        leftover_highlights = leftover_exp_highlights

                    HighlightCoverageGrader._increase_last_leftover_range_by_one(
                        index, leftover_highlights
                    )

        if current_range:
            overlapping_indices.append(current_range)

        return overlapping_indices, leftover_gen_highlights, leftover_exp_highlights

    @staticmethod
    def _increase_current_range_by_one(
        current_range: IndexRange | None, index: int
    ) -> IndexRange:
        if current_range is None:
            return IndexRange(index, index + 1)
        return IndexRange(current_range.start, index + 1)

    @staticmethod
    def _increase_last_leftover_range_by_one(
        index: int, leftover_highlights: _HighlightRange
    ) -> _HighlightRange:
        if leftover_highlights and leftover_highlights[-1].stop == index:
            leftover_highlights[-1] = (
                HighlightCoverageGrader._increase_current_range_by_one(
                    leftover_highlights[-1], index
                )
            )
        else:
            leftover_highlights.append(IndexRange(index, index + 1))

        return leftover_highlights
