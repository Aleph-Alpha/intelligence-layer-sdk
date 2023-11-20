from typing import Iterable

from intelligence_layer.core import (
    Evaluator,
    Example,
    NoOpTracer,
    SequenceDataset,
    Tracer,
)


class TestTaskEvaluator(Evaluator[None, None, None, None]):
    def do_evaluate(self, input: None, tracer: Tracer, expected_output: None) -> None:
        raise RuntimeError("Test Exception")

    def aggregate(self, evaluations: Iterable[None]) -> None:
        return None


def test_evaluate_dataset_does_not_raise_exception_on_failed_example() -> None:
    evaluator = TestTaskEvaluator()
    dataset = SequenceDataset(
        name="test",
        examples=[Example(input=None, expected_output=None)],
    )
    evaluator.evaluate_dataset(dataset, NoOpTracer())
