

from intelligence_layer.evaluation.dataset.data_platform_dataset_repository import DataPlatformDatasetRepository
from intelligence_layer.evaluation.dataset.domain import Example



def test_all() -> None:
    repository = DataPlatformDatasetRepository()
    examples = sorted([Example(input="test-input", expected_output="test-output"), Example(input="test-input", expected_output="test-output")], key= lambda example: example.id)

    dataset = repository.create_dataset(examples, "ignored")
    returned_examples = repository.examples(dataset.id, str, str)
    assert examples == returned_examples

    ids = repository.dataset_ids()
    assert dataset.id in ids

    retrieved_dataset = repository.dataset(dataset.id)
    assert retrieved_dataset is not None
    assert retrieved_dataset.id == dataset.id
    #assert retrieved_dataset.name == dataset.name


def test_ids() -> None:
    repository = DataPlatformDatasetRepository()
    ids = repository.dataset_ids()
    list_ids = list(ids)
    assert len(list_ids) > 0
    print(list_ids)