
from threading import Lock
from typing import Mapping, Sequence
import nltk  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from rouge import Rouge  # type: ignore
from nltk.translate.bleu_score import sentence_bleu  # type: ignore
from dataclasses import dataclass

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


class BleuGrader():
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


class RougeGrader():
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
