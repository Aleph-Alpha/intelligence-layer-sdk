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
        overview = self._aggregation_overviews.get(aggregation_id, None)
        if overview is None or type(overview.statistics) is aggregation_type:
            return overview
        return AggregationOverview[AggregatedEvaluation](
            evaluation_overviews=overview.evaluation_overviews,
            id=overview.id,
            start=overview.start,
            end=overview.end,
            successful_evaluation_count=overview.successful_evaluation_count,
            crashed_during_evaluation_count=overview.crashed_during_evaluation_count,
            description=overview.description,
            statistics=aggregation_type.model_validate(overview.statistics),
            labels=overview.labels,
            metadata=overview.metadata,
        )

    def aggregation_overview_ids(self) -> Sequence[str]:
        return sorted(list(self._aggregation_overviews.keys()))
