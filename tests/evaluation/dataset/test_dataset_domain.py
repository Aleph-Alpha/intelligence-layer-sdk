from intelligence_layer.evaluation import Dataset


def test_default_values_are_set() -> None:
    dataset = Dataset(name="Test")

    assert dataset.id is not None
    assert len(dataset.metadata) == 0
    assert len(dataset.labels) == 0


def test_default_values_are_not_changed() -> None:
    modified_dataset = Dataset(name="Modified Dataset")
    modified_dataset.labels.add("test_label")
    modified_dataset.metadata.update({"key": "value"})

    default_dataset = Dataset(name="Default Dataset")

    assert modified_dataset.labels != default_dataset.labels
    assert modified_dataset.metadata != default_dataset.metadata
