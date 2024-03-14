from datasets import load_dataset

from intelligence_layer.evaluation.data_storage.single_huggingface_dataset_repository import (
    SingleHuggingfaceDatasetRepository,
    MultipleChoiceInput,
)


def test_load_example_for_existing_dataset() -> None:

    dataset = load_dataset(
        path="hails/mmlu_no_train", name="all", trust_remote_code=True
    )
    repository = SingleHuggingfaceDatasetRepository[MultipleChoiceInput, str](dataset)

    examples = repository.examples(
        dataset_id="", input_type=MultipleChoiceInput, expected_output_type=str
    )

    first_example = next(iter(examples))
    assert first_example.input.question != ""
