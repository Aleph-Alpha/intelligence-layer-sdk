from datasets import Dataset, DatasetDict  # type: ignore
from pytest import fixture

from intelligence_layer.evaluation import (
    MultipleChoiceInput,
    SingleHuggingfaceDatasetRepository,
)


@fixture
def mmlu_test_example() -> DatasetDict:
    # this is the first example of the hails/mmlu dataset.
    # The metadata is not completely correct on the DatasetDict, but enough for the current case
    data = DatasetDict(
        {
            "test": Dataset.from_list(
                [
                    {
                        "question": "Which of the following best describes the balance the Supreme Court has struck between the establishment clause and the free-exercise clause?",
                        "subject": "high_school_government_and_politics",
                        "choices": [
                            'Freedom of speech is protected except in certain situations, such as yelling "fire" in a crowded theater.',
                            "Once a church has been recognized by the federal government, its tax-exempt status can never be revoked.",
                            "Once Congress has created an administrative agency, that agency can be dissolved only by a constitutional amendment.",
                            "State-sponsored prayer during school hours is prohibited, but voluntary prayer by student groups before school is allowed.",
                        ],
                        "answer": 3,
                    }
                ]
            ),
            "validation": Dataset.from_list(
                [
                    {
                        "question": "Which of the following situations does NOT occur in a federal state?",
                        "subject": "high_school_geography",
                        "choices": [
                            "Central government possesses a two-level system of government.",
                            "Central government governs country as a single unit.",
                            "It often possesses a written constitution.",
                            "Lower-level divisions have unique powers.",
                        ],
                        "answer": 1,
                    }
                ]
            ),
            "dev": Dataset.from_list(
                [
                    {
                        "question": "Box a nongovernmental not-for-profit organization had the following transactions during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported as net cash provided by financing activities in Box's statement of cash flows?",
                        "subject": "professional_accounting",
                        "choices": ["$70,000", "$75,000", "$80,000", "100000"],
                        "answer": 3,
                    }
                ]
            ),
        }
    )
    return data


def test_load_example_for_existing_dataset(mmlu_test_example: DatasetDict) -> None:
    dataset = mmlu_test_example
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
