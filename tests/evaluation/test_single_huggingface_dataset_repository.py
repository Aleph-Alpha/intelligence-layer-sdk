from datasets import load_dataset  # type: ignore

from intelligence_layer.evaluation import (
    MultipleChoiceInput,
    SingleHuggingfaceDatasetRepository,
)


def test_load_example_for_existing_dataset() -> None:

    dataset = load_dataset(
        path="hails/mmlu_no_train", name="all", trust_remote_code=True
    )
    repository = SingleHuggingfaceDatasetRepository(dataset)

    examples = repository.examples(
        dataset_id="", input_type=MultipleChoiceInput, expected_output_type=str
    )

    first_example = next(iter(examples))
    assert (
        first_example.input.question
        == "Which of the following best describes the balance the Supreme Court has struck between the establishment clause and the free-exercise clause?"
    )

    assert first_example.input.choices == [
        'Freedom of speech is protected except in certain situations, such as yelling "fire" in a crowded theater.',
        "Once a church has been recognized by the federal government, its tax-exempt status can never be revoked.",
        "Once Congress has created an administrative agency, that agency can be dissolved only by a constitutional amendment.",
        "State-sponsored prayer during school hours is prohibited, but voluntary prayer by student groups before school is allowed.",
    ]

    assert first_example.expected_output == "D"

    assert first_example.id == "0"

    first_example_by_id = repository.example(
        dataset_id="",
        example_id=str(0),
        input_type=MultipleChoiceInput,
        expected_output_type=str,
    )

    assert first_example == first_example_by_id