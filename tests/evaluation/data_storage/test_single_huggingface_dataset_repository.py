from datasets import load_dataset

from intelligence_layer.evaluation.data_storage.single_huggingface_dataset_repository import (
    SingleHuggingfaceDatasetRepository,
)


def test_load_example_for_existing_dataset() -> None:
    repository = SingleHuggingfaceDatasetRepository(
        load_dataset(path="hails/mmlu_no_train", name="all")
    )
