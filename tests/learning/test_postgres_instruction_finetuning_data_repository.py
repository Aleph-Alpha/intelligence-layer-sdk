from intelligence_layer.learning import (
    InstructionFinetuningSample,
    PostgresInstructionFinetuningDataRepository,
    RawInstructionFinetuningSample,
)


def test_postgres_instruction_finetuning_data_repository_can_store_load_and_delete_sample(
    postgres_instruction_finetuning_data_repository: PostgresInstructionFinetuningDataRepository,
    instruction_finetuning_sample: InstructionFinetuningSample,
) -> None:
    postgres_instruction_finetuning_data_repository.store_sample(
        instruction_finetuning_sample
    )
    loaded_sample = postgres_instruction_finetuning_data_repository.sample(
        instruction_finetuning_sample.id
    )

    assert instruction_finetuning_sample == loaded_sample

    postgres_instruction_finetuning_data_repository.delete_sample(
        instruction_finetuning_sample.id
    )
    no_sample_expected = postgres_instruction_finetuning_data_repository.sample(
        instruction_finetuning_sample.id
    )

    assert no_sample_expected is None


def test_postgres_instruction_finetuning_data_repository_can_store_load_and_delete_samples(
    postgres_instruction_finetuning_data_repository: PostgresInstructionFinetuningDataRepository,
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
) -> None:
    samples = [
        InstructionFinetuningSample.from_raw_sample(raw_instruction_finetuning_sample)
        for _ in range(10)
    ]
    ids = [sample.id for sample in samples]

    postgres_instruction_finetuning_data_repository.store_samples(samples)
    loaded_samples = postgres_instruction_finetuning_data_repository.samples(ids)

    assert set(ids) == set(loaded_sample.id for loaded_sample in loaded_samples)

    postgres_instruction_finetuning_data_repository.delete_samples(ids)
    no_samples_expected = postgres_instruction_finetuning_data_repository.samples(ids)

    assert list(no_samples_expected) == []


def test_postgres_instruction_finetuning_data_repository_can_show_first_n_samples(
    postgres_instruction_finetuning_data_repository: PostgresInstructionFinetuningDataRepository,
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
) -> None:
    n = 10
    samples = [
        InstructionFinetuningSample.from_raw_sample(raw_instruction_finetuning_sample)
        for _ in range(n)
    ]

    postgres_instruction_finetuning_data_repository.store_samples(samples)
    head = list(postgres_instruction_finetuning_data_repository.head(n // 2))

    assert len(head) == n // 2
