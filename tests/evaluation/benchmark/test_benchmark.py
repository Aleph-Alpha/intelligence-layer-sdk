
from intelligence_layer.evaluation.benchmark.studio_benchmark import (
    StudioBenchmarkRepository,
    create_aggregation_logic_identifier,
    create_evaluation_logic_identifier,
)
from intelligence_layer.evaluation.dataset.studio_dataset_repository import (
    StudioDatasetRepository,
)
from tests.evaluation.conftest import (
    DummyAggregation,
    DummyAggregationLogic,
    DummyEvaluation,
    DummyEvaluationLogic,
)


def test_extract_types_from_eval_logic() -> None:
    eval_logic = DummyEvaluationLogic()
    created_identifier = create_evaluation_logic_identifier(eval_logic)
    dummy_logic = "if output == FAIL_IN_EVAL_INPUT:"
    dummy_type = "DummyEvaluation"
    assert dummy_logic in created_identifier.logic
    assert dummy_type in created_identifier.logic

    assert created_identifier.input_schema["type"] == "string"
    assert created_identifier.expected_output_schema["type"] == "null"

    assert created_identifier.evaluation_schema["type"] == "object"
    assert (
        created_identifier.evaluation_schema["properties"]["result"]["type"] == "string"
    )


def test_extract_types_from_aggregation_logic() -> None:
    aggregation_logic = DummyAggregationLogic()
    created_identifier = create_aggregation_logic_identifier(aggregation_logic)
    dummy_logic = "return DummyAggregation(num_evaluations=len(list(evaluations)))"

    assert dummy_logic in created_identifier.logic
    assert created_identifier.evaluation_schema["type"] == "object"
    assert created_identifier.aggregation_schema["type"] == "object"

def test_create_benchmark(studio_benchmark_repository: StudioBenchmarkRepository[str, str, None, DummyEvaluation, DummyAggregation], studio_dataset_repository: StudioDatasetRepository) -> None:
    dataset_id = studio_dataset_repository.create_dataset(examples=[], dataset_name="dataset").id
    eval_logic = DummyEvaluationLogic()
    aggregation_logic = DummyAggregationLogic()
    benchmark = studio_benchmark_repository.create_benchmark(dataset_id, eval_logic, aggregation_logic, "benchmark_name")

    assert benchmark.id


