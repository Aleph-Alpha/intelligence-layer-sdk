import pytest

from intelligence_layer.evaluation.aggregation.domain import AggregationOverview
from intelligence_layer.evaluation.evaluation.domain import EvaluationFailed
from tests.evaluation.conftest import DummyAggregatedEvaluation


def test_raise_on_exception_for_evaluation_run_overview(
    aggregation_overview: AggregationOverview[DummyAggregatedEvaluation],
) -> None:
    with pytest.raises(EvaluationFailed):
        aggregation_overview.raise_on_evaluation_failure()
