import langdetect  # type: ignore
from pytest import fixture

from intelligence_layer.evaluation import LanguageMatchesGrader, HighlightCoverageGrader


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


def test_highlight_coverage_grader_returns_perfect_score_for_good_highlights() -> None:
    grader = HighlightCoverageGrader()
    text = "Hello, I am working on this highlight grader."
    generated_highlights = {text[5:7], text[7:18]}
    expected_highlights = {text[4:8], text[8:10], text[13:17]}
    
    assert grader.compare_highlights(
        text=text,
        generated_highlights=generated_highlights,
        expected_highlights=expected_highlights
    ) == 1.0


def test_highlight_coverage_grader_returns_null_score_for_all_text_highlighted() -> None:
    grader = HighlightCoverageGrader()
    text = "Hello, I am working on this highlight grader."
    generated_highlights = {text}
    expected_highlights = {text[5:10]}
    
    assert grader.compare_highlights(
        text=text,
        generated_highlights=generated_highlights,
        expected_highlights=expected_highlights
    ) == 0.0
