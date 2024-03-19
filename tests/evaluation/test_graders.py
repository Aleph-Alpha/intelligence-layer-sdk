import langdetect  # type: ignore
import nltk
from pytest import fixture

from intelligence_layer.evaluation.evaluation.graders import LanguageMatchesGrader


@fixture(scope="session")
def set_deterministic_seed() -> None:
    langdetect.DetectorFactory.seed = 0


def test_language_matches_grader_correctly_detects_languages_match() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match("This is a test.", "This is also a test.")


def test_language_matches_grader_correctly_detects_languages_dont_match() -> None:
    grader = LanguageMatchesGrader()
    assert not grader.languages_match("This is a test.", "Dies ist noch ein Test.")


def test_language_matches_grader_returns_true_if_input_language_is_unclear() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match("This is a test. Das ist ein Test.", "")


def test_language_matches_grader_returns_false_if_output_language_is_unclear() -> None:
    grader = LanguageMatchesGrader()
    assert not grader.languages_match(
        "This is a test.", "This is a test. Das ist ein Test."
    )
