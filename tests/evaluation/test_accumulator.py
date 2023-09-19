from intelligence_layer.evaluation import MeanAccumulator


def test_mean_accumulator_returns_mean() -> None:
    acc = MeanAccumulator()
    assert acc.extract() == 0.0
    acc.add(1)
    assert acc.extract() == 1.0
    acc.add(0)
    assert acc.extract() == 0.5


def test_mean_accumulator_returns_stdev_and_se() -> None:
    acc = MeanAccumulator()
    assert acc.standard_deviation() == 0.0
    assert acc.standard_error() == 0.0
    acc.add(1)
    assert acc.standard_deviation() == 0.0
    assert acc.standard_error() == 0.0
    acc.add(0)
    assert acc.standard_deviation() == 0.5
    assert round(acc.standard_error(), 3) == 0.354
