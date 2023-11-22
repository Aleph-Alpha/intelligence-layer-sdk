from typing import Iterable, Optional

from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import (
    EvaluationException,
    Evaluator,
    Example,
    InMemoryEvaluationRepository,
    NoOpTracer,
    SequenceDataset,
    Tracer,
    Dataset,
    TaskTrace
)


class DummyEvaluation(BaseModel):
    result: str


class DummyEvaluator(Evaluator[Optional[str], None, DummyEvaluation, None]):
    def do_evaluate(
        self, input: Optional[str], tracer: Tracer, expected_output: None
    ) -> DummyEvaluation:
        if input is None:
            return DummyEvaluation(result="pass") 
        raise RuntimeError(input) 


    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> None:
        return None


@fixture
def evaluation_repository() -> InMemoryEvaluationRepository:
    return InMemoryEvaluationRepository()

@fixture
def dummy_evaluator(evaluation_repository: InMemoryEvaluationRepository) -> DummyEvaluator:
    return DummyEvaluator(evaluation_repository)
    

def test_evaluate_dataset_does_not_throw_an_exception_for_failure(dummy_evaluator: DummyEvaluator) -> None:
    dataset: Dataset[Optional[str], None] = SequenceDataset(
        name="test",
        examples=[Example(input="fail", expected_output=None) ],
    )
    dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())

    
def test_evaluate_dataset_stores_example_results(dummy_evaluator: DummyEvaluator) -> None:
    evaluation_repository = dummy_evaluator.repository
    dataset: Dataset[Optional[str], None] = SequenceDataset(
        name="test",
        examples=[Example(input=None, expected_output=None), Example(input="fail", expected_output=None) ],
    )

    evaluation_run_overview = dummy_evaluator.evaluate_dataset(dataset, NoOpTracer())
    results = evaluation_repository.evaluation_run_results(
        evaluation_run_overview.id, DummyEvaluation
    )

    assert isinstance(results[0].result, DummyEvaluation)
    assert isinstance(results[1].result, EvaluationException)
    # assert [isinstance(r.example_trace, TaskTrace) for r in results]
    assert len(results) == len(dataset.examples)
    
