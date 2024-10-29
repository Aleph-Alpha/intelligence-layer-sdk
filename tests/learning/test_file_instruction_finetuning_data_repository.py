import random

from intelligence_layer.core import Language, Message
from intelligence_layer.learning import (
    FileInstructionFinetuningDataRepository,
    InstructionFinetuningSample,
    InstructionFinetuningSample_,
    InstructionFinetuningSampleAttributes,
    RawInstructionFinetuningSample,
)


def test_file_instruction_finetuning_data_repository_can_store_load_and_delete_sample(
    file_instruction_finetuning_data_repository: FileInstructionFinetuningDataRepository,
    instruction_finetuning_sample: InstructionFinetuningSample,
) -> None:
    file_instruction_finetuning_data_repository.store_sample(
        instruction_finetuning_sample
    )
    loaded_sample = file_instruction_finetuning_data_repository.sample(
        instruction_finetuning_sample.id
    )

    assert instruction_finetuning_sample == loaded_sample

    file_instruction_finetuning_data_repository.delete_sample(
        instruction_finetuning_sample.id
    )
    no_sample_expected = file_instruction_finetuning_data_repository.sample(
        instruction_finetuning_sample.id
    )

    assert no_sample_expected is None


def test_file_instruction_finetuning_data_repository_can_store_load_and_delete_samples(
    file_instruction_finetuning_data_repository: FileInstructionFinetuningDataRepository,
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
) -> None:
    samples = [
        InstructionFinetuningSample.from_raw_sample(raw_instruction_finetuning_sample)
        for _ in range(10)
    ]
    ids = [sample.id for sample in samples]

    file_instruction_finetuning_data_repository.store_samples(samples)
    loaded_samples = file_instruction_finetuning_data_repository.samples(ids)

    assert set(ids) == set(loaded_sample.id for loaded_sample in loaded_samples)

    file_instruction_finetuning_data_repository.delete_samples(ids)
    no_samples_expected = file_instruction_finetuning_data_repository.samples(ids)

    assert list(no_samples_expected) == []


def test_file_instruction_finetuning_data_repository_can_show_first_n_samples(
    file_instruction_finetuning_data_repository: FileInstructionFinetuningDataRepository,
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
) -> None:
    n = 10
    samples = [
        InstructionFinetuningSample.from_raw_sample(raw_instruction_finetuning_sample)
        for _ in range(n)
    ]

    file_instruction_finetuning_data_repository.store_samples(samples)
    head = list(file_instruction_finetuning_data_repository.head(n // 2))

    assert len(head) == n // 2


def test_file_instruction_finetuning_data_repository_can_return_sample_with_filter(
    file_instruction_finetuning_data_repository: FileInstructionFinetuningDataRepository,
) -> None:
    expected_minimum_quality = 2
    expected_maximum_quality = 4
    expected_domain = "specific domain"
    sample = InstructionFinetuningSample(
        messages=[
            Message(role="user", content="Hi. Tell me a weird German word."),
            Message(
                role="assistant",
                content='Do you know what a "Donaudampfschifffahrtsgesellschaft" is?',
            ),
        ],
        attributes=InstructionFinetuningSampleAttributes(
            source="example",
            domain=expected_domain,
            quality=random.randint(expected_minimum_quality, expected_maximum_quality),
            languages=[Language("en")],
        ),
        external_id="example_1",
    )
    filter_expression = (
        InstructionFinetuningSample_.quality >= expected_minimum_quality
    ) & (InstructionFinetuningSample_.quality <= expected_maximum_quality)

    file_instruction_finetuning_data_repository.store_sample(sample)
    filtered_samples = list(
        file_instruction_finetuning_data_repository.samples_with_filter(
            filter_expression=filter_expression, limit=1
        )
    )

    assert len(filtered_samples) == 1

    filtered_sample = filtered_samples[0]
    quality = filtered_sample.attributes.quality

    assert quality
    assert expected_minimum_quality <= quality <= expected_maximum_quality

    domain = filtered_sample.attributes.domain

    assert domain
    assert domain == expected_domain
