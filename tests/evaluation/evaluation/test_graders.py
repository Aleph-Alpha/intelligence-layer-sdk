from intelligence_layer.evaluation import HighlightCoverageGrader, LanguageMatchesGrader


def test_language_matches_grader_correctly_detects_languages_match() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match(
        "This is a test of a sentence in one language.",
        "This is also a test of another sentence in the same language.",
    )


def test_language_matches_grader_correctly_detects_languages_dont_match() -> None:
    grader = LanguageMatchesGrader()
    assert not grader.languages_match(
        "This is a test of a sentence in one language.",
        "Dies ist ein weiterer Test eines anderen Satzes in einer anderen Sprache.",
    )


def test_language_matches_grader_returns_true_if_input_language_is_unclear() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match(
        "This is a test of a sentence in one language. Das ist ein Test eines anderen Satzes in einer anderen Sprache.",
        "",
    )


def test_language_matches_grader_returns_false_if_output_language_is_unclear() -> None:
    grader = LanguageMatchesGrader()
    assert not grader.languages_match(
        "This is a test of a sentence in one language.",
        "This is a test of a sentence in one language. Das ist ein Test eines anderen Satzes in einer anderen Sprache.",
    )


def test_language_matches_grader_can_handle_difficult_input() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match(
        "1. This test is difficult. 2. And it has esp. diff. characters.",
        "a) Here is a another test.\nb) How can it handle enumerations?",
    )


def test_language_matches_grader_empty_input_and_output() -> None:
    grader = LanguageMatchesGrader()
    assert grader.languages_match(
        "",
        "",
    )


def test_highlight_coverage_grader_returns_perfect_score_if_exact_match() -> None:
    grader = HighlightCoverageGrader()
    generated_highlights = [(3, 5)]
    expected_highlights = [(3, 5)]

    assert (
        grader.compute_fscores(
            generated_highlight_indices=generated_highlights,
            expected_highlight_indices=expected_highlights,
        ).f_score
        == 1.0
    )


def test_highlight_coverage_grader_returns_worst_score_if_no_expected_highlight_identified() -> (
    None
):
    grader = HighlightCoverageGrader()
    generated_highlights = [(0, 3)]
    expected_highlights = [(3, 5)]

    assert (
        grader.compute_fscores(
            generated_highlight_indices=generated_highlights,
            expected_highlight_indices=expected_highlights,
        ).f_score
        == 0.0
    )


def test_highlight_coverage_grader_returns_perfect_score_if_highlights_exactly_identified_but_split_up() -> (
    None
):
    grader = HighlightCoverageGrader()
    generated_highlights = [(3, 4), (4, 7)]
    expected_highlights = [(3, 7)]

    assert (
        grader.compute_fscores(
            generated_highlight_indices=generated_highlights,
            expected_highlight_indices=expected_highlights,
        ).f_score
        == 1.0
    )


def test_highlight_coverage_grader_returns_perfect_score_if_highlights_exactly_identified_but_merged() -> (
    None
):
    grader = HighlightCoverageGrader()
    generated_highlights = [(3, 6)]
    expected_highlights = [(3, 5), (5, 6)]

    assert (
        grader.compute_fscores(
            generated_highlight_indices=generated_highlights,
            expected_highlight_indices=expected_highlights,
        ).f_score
        == 1.0
    )


def test_highlight_coverage_grader_returns_perfect_score_if_all_but_more_highlights_identified_and_beta_equals_zero() -> (
    None
):
    grader = HighlightCoverageGrader(beta_factor=0)
    generated_highlights = [(3, 5)]
    expected_highlights = [(3, 6)]

    assert (
        grader.compute_fscores(
            generated_highlight_indices=generated_highlights,
            expected_highlight_indices=expected_highlights,
        ).f_score
        == 1.0
    )


def test_highlight_coverage_grader_returns_nonzero_nonperfect_score_with_some_false_negatives_and_some_false_positives() -> (
    None
):
    grader = HighlightCoverageGrader()
    generated_highlights = [(3, 5)]
    expected_highlights = [(4, 7)]

    assert (
        grader.compute_fscores(
            generated_highlight_indices=generated_highlights,
            expected_highlight_indices=expected_highlights,
        ).f_score
        == 0.4
    )
