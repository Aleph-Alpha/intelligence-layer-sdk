import langdetect  # type: ignore
from pytest import fixture, mark

from intelligence_layer.evaluation.evaluation.graders import LanguageMatchesGrader


@fixture(scope="session")
def set_deterministic_seed() -> None:
    langdetect.DetectorFactory.seed = 0


@mark.skip("nltk download/load does not yet properly work in GitHub pipeline")
def test_language_matches_grader_correctly_detects_languages_match() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match("This is a test.", "This is also a test.")


@mark.skip("nltk download/load does not yet properly work in GitHub pipeline")
def test_language_matches_grader_correctly_detects_languages_dont_match() -> None:
    grader = LanguageMatchesGrader()
    assert not grader.languages_match("This is a test.", "Dies ist noch ein Test.")


@mark.skip("nltk download/load does not yet properly work in GitHub pipeline")
def test_language_matches_grader_returns_true_if_input_language_is_unclear() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match("This is a test. Das ist ein Test.", "")


@mark.skip("nltk download/load does not yet properly work in GitHub pipeline")
def test_language_matches_grader_returns_false_if_output_language_is_unclear() -> None:
    grader = LanguageMatchesGrader()
    assert not grader.languages_match(
        "This is a test.", "This is a test. Das ist ein Test."
    )
