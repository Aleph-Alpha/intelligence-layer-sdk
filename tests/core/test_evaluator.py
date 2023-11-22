from typing import Iterable

from pydantic import BaseModel

from intelligence_layer.core import (
    EvaluationException,
    Evaluator,
    Example,
    InMemoryEvaluationRepository,
    NoOpTracer,
    SequenceDataset,
    Tracer,
)


class DummyEvaluation(BaseModel):
    result: str


class DummyEvaluator(Evaluator[None, None, DummyEvaluation, None]):
    def do_evaluate(
        self, input: None, tracer: Tracer, expected_output: None
    ) -> DummyEvaluation:
        raise RuntimeError("Test Exception")

    def aggregate(self, evaluations: Iterable[DummyEvaluation]) -> None:
        return None


def test_evaluate_dataset_does_not_raise_exception_on_failed_example() -> None:
    evaluation_repository = InMemoryEvaluationRepository()
    evaluator = DummyEvaluator(evaluation_repository)
    dataset = SequenceDataset(
        name="test",
        examples=[Example(input=None, expected_output=None)],
    )
    evaluation_run_overview = evaluator.evaluate_dataset(dataset, NoOpTracer())
    results = evaluation_repository.evaluation_run_results(
        evaluation_run_overview.id, DummyEvaluation
    )
    assert all(isinstance(result.result, EvaluationException) for result in results)
    assert len(results) == len(dataset.examples)
