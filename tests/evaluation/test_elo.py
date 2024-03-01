from pydantic import BaseModel
from intelligence_layer.evaluation import MatchOutcome


class MatchOutcomeModel(BaseModel):
    match_outcome: MatchOutcome


def test_match_outcome_serializes() -> None:
    match_outcome_model = MatchOutcomeModel(match_outcome=MatchOutcome.A_WINS)
    dumped = match_outcome_model.model_dump_json()
    loaded = MatchOutcomeModel.model_validate_json(dumped)

    assert loaded == match_outcome_model
