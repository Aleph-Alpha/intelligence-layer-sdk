from intelligence_layer.evaluation import MeanAccumulator


def test_mean_accumulator() -> None:
    acc = MeanAccumulator()
    assert acc.extract() == 0.0
    acc.add(1)
    assert acc.extract() == 1.0
    acc.add(0)
    assert acc.extract() == 0.5
