from collections.abc import Sequence
from typing import Any, Optional

from intelligence_layer.evaluation.aggregation.aggregation_repository import (
    AggregationRepository,
)
from intelligence_layer.evaluation.aggregation.domain import (
    AggregatedEvaluation,
    AggregationOverview,
)


class InMemoryAggregationRepository(AggregationRepository):
    def __init__(self) -> None:
        super().__init__()
        self._aggregation_overviews: dict[str, AggregationOverview[Any]] = dict()

    def store_aggregation_overview(
        self, aggregation_overview: AggregationOverview[AggregatedEvaluation]
    ) -> None:
        self._aggregation_overviews[aggregation_overview.id] = aggregation_overview

    def aggregation_overview(
        self, aggregation_id: str, aggregation_type: type[AggregatedEvaluation]
    ) -> Optional[AggregationOverview[AggregatedEvaluation]]:
        return self._aggregation_overviews.get(aggregation_id, None)

    def aggregation_overview_ids(self) -> Sequence[str]:
        return sorted(list(self._aggregation_overviews.keys()))
