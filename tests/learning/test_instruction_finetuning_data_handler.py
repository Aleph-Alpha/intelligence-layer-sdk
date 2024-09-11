from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import and_

from intelligence_layer.core import Message
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.model import Pharia1ChatModel
from intelligence_layer.learning import (
    EnrichAction,
    InstructionFinetuningDataHandler,
    InstructionFinetuningSample,
    InstructionFinetuningSampleAttributes,
    InvalidSampleError,
    RawInstructionFinetuningSample,
)
from intelligence_layer.learning.models import InstructionFinetuningSample_


def test_instruction_finetuning_data_handler_raises_on_sample_with_no_content(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    sample = RawInstructionFinetuningSample(
        messages=[
            Message(role="user", content="\n"),
            Message(role="assistant", content="Your message was empty."),
        ],
        attributes=InstructionFinetuningSampleAttributes(source="example"),
    )

    with pytest.raises(InvalidSampleError, match=" does not contain anything."):
        instruction_finetuning_data_handler._validate_sample(sample)


def test_instruction_finetuning_data_handler_raises_on_sample_that_starts_with_assistant_message(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    sample = RawInstructionFinetuningSample(
        messages=[
            Message(
                role="assistant",
                content="As the assistant, I shoulnd't send the first message.",
            ),
        ],
        attributes=InstructionFinetuningSampleAttributes(source="example"),
    )

    with pytest.raises(InvalidSampleError, match=" starts with an assistant message."):
        instruction_finetuning_data_handler._validate_sample(sample)


def test_instruction_finetuning_data_handler_raises_on_sample_with_consecutive_messages_by_same_role(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    sample = RawInstructionFinetuningSample(
        messages=[
            Message(role="user", content="Some message by any role..."),
            Message(role="user", content="... shouldn't be followed by another."),
        ],
        attributes=InstructionFinetuningSampleAttributes(source="example"),
    )

    with pytest.raises(InvalidSampleError, match=" messages right after another."):
        instruction_finetuning_data_handler._validate_sample(sample)


def test_instruction_finetuning_data_handler_raises_on_sample_with_no_messages(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    sample = RawInstructionFinetuningSample(
        messages=[], attributes=InstructionFinetuningSampleAttributes(source="example")
    )

    with pytest.raises(InvalidSampleError, match=" has less than 2 messages."):
        instruction_finetuning_data_handler._validate_sample(sample)


def test_instruction_finetuning_data_handler_raises_on_sample_with_multiple_system_messages(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    sample = RawInstructionFinetuningSample(
        messages=[
            Message(role="system", content="Some message..."),
            Message(role="user", content="Some other message..."),
            Message(role="system", content="A second system message."),
        ],
        attributes=InstructionFinetuningSampleAttributes(source="example"),
    )

    with pytest.raises(InvalidSampleError, match=" has multiple system messages."):
        instruction_finetuning_data_handler._validate_sample(sample)


def test_instruction_finetuning_data_handler_raises_on_sample_with_no_assistant_message(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    sample = RawInstructionFinetuningSample(
        messages=[
            Message(role="system", content="Some message..."),
            Message(role="user", content="Some other message..."),
        ],
        attributes=InstructionFinetuningSampleAttributes(source="example"),
    )

    with pytest.raises(InvalidSampleError, match=" has no assistant messages."):
        instruction_finetuning_data_handler._validate_sample(sample)


def test_instruction_finetuning_data_handler_raises_on_sample_that_ends_on_user_message(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    sample = RawInstructionFinetuningSample(
        messages=[
            Message(role="system", content="Some message..."),
            Message(role="assistant", content="Some answer..."),
            Message(role="user", content="A user message should never be the last."),
        ],
        attributes=InstructionFinetuningSampleAttributes(source="example"),
    )

    with pytest.raises(InvalidSampleError, match=" ends on a user message."):
        instruction_finetuning_data_handler._validate_sample(sample)


def test_instruction_finetuning_data_handler_can_add_and_update_sample(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
) -> None:
    sample_id = instruction_finetuning_data_handler.add_sample(
        raw_instruction_finetuning_sample
    )

    assert sample_id

    sample = instruction_finetuning_data_handler.sample(sample_id)

    assert sample

    instruction_finetuning_data_handler.update_sample(
        sample_id, language_action=EnrichAction.REMOVE
    )
    updated_sample = instruction_finetuning_data_handler.sample(sample_id)

    assert updated_sample
    assert sample.attributes.languages
    assert not updated_sample.attributes.languages


def test_instruction_finetuning_data_handler_can_store_and_load_samples_in_batch(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
) -> None:
    samples = [raw_instruction_finetuning_sample for _ in range(5)]
    domain_action = EnrichAction.SKIP
    quality_action = EnrichAction.SKIP
    language_action = EnrichAction.SKIP

    added_ids = list(
        instruction_finetuning_data_handler.add_samples(
            samples, domain_action, quality_action, language_action
        )
    )
    loaded_samples = [
        instruction_finetuning_data_handler.sample(id) for id in added_ids
    ]

    assert len(samples) == len(added_ids) == len(loaded_samples)
    assert set(added_ids) == set(sample.id for sample in loaded_samples if sample)


def test_instruction_finetuning_data_handler_can_enrich_sample(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
    instruction_finetuning_sample: InstructionFinetuningSample,
) -> None:
    enriched_sample = instruction_finetuning_data_handler._enrich_sample(
        instruction_finetuning_sample,
        domain_action=EnrichAction.REMOVE,
        quality_action=EnrichAction.SKIP,
        language_action=EnrichAction.GET,
    )

    assert enriched_sample.attributes.domain is None
    assert (
        enriched_sample.attributes.quality
        == instruction_finetuning_sample.attributes.quality
    )
    assert isinstance(enriched_sample.attributes.languages, list)


def test_instruction_finetuning_data_handler_can_return_samples_with_filter(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
) -> None:
    expected_quality = 1
    expected_domain = "general"
    bad_sample = RawInstructionFinetuningSample(
        messages=[
            Message(role="user", content="Hi."),
            Message(role="assistant", content="I don't want to help you."),
        ],
        attributes=InstructionFinetuningSampleAttributes(
            source="example",
            domain=expected_domain,
            quality=expected_quality,
            languages=[Language("en")],
        ),
        external_id="example_1",
    )

    sample_id = instruction_finetuning_data_handler.add_sample(bad_sample)
    filter_expression = and_(
        InstructionFinetuningSample_.quality == expected_quality,
        InstructionFinetuningSample_.domain == expected_domain,
    )
    filtered_samples = list(
        instruction_finetuning_data_handler.samples_with_filter(filter_expression, 1000)
    )

    assert sample_id in [sample.id for sample in filtered_samples]
    assert all(
        sample.attributes.quality == expected_quality for sample in filtered_samples
    )
    assert all(
        sample.attributes.domain == expected_domain for sample in filtered_samples
    )


def test_instruction_finetuning_data_handler_can_create_train_set(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
    instruction_finetuning_sample: InstructionFinetuningSample,
    pharia_1_chat_model: Pharia1ChatModel,
) -> None:
    long_context_sample = InstructionFinetuningSample(
        messages=[
            Message(role="user", content="Hi."),
            Message(role="assistant", content="Hello, how can I help you?" * 2000),
        ],
        attributes=InstructionFinetuningSampleAttributes(
            source="example", domain="general", quality=5, languages=[Language("en")]
        ),
        external_id="example_1",
    )
    samples = [instruction_finetuning_sample, long_context_sample]

    with pytest.warns() as record:
        train_set = instruction_finetuning_data_handler.samples_to_train_set(
            pharia_1_chat_model, samples
        )

    assert long_context_sample.id in str(record[0].message)
    assert (
        str(record[1].message) == "Emitted 1 sample(s) due to context size constraints."
    )

    assert len(train_set.data) == len(train_set.ids) == 1

    train_sample = train_set.data[0]

    assert not train_sample[0].has_loss
    assert train_sample[1].has_loss
    assert all(item.type == "text" for item in train_sample)
    assert instruction_finetuning_sample.id == train_set.ids[0]


def test_instruction_finetuning_data_handler_can_compile_train_set_to_file(
    instruction_finetuning_data_handler: InstructionFinetuningDataHandler,
    raw_instruction_finetuning_sample: RawInstructionFinetuningSample,
    pharia_1_chat_model: Pharia1ChatModel,
    tmp_path: Path,
) -> None:
    num_samples = 10
    instruction_finetuning_data_handler.add_samples(
        (raw_instruction_finetuning_sample for _ in range(num_samples)),
        EnrichAction.SKIP,
        EnrichAction.SKIP,
        EnrichAction.SKIP,
    )
    instruction_finetuning_data_handler.compile_train_set(
        tmp_path, pharia_1_chat_model, limit=num_samples
    )

    train_set = instruction_finetuning_data_handler._read_json_or_jsonl(
        tmp_path / "train_set.jsonl"
    )
    ids = instruction_finetuning_data_handler._read_json_or_jsonl(
        tmp_path / "ids.jsonl"
    )
    statistics: Mapping[Any, Mapping[Any, int]] = (
        instruction_finetuning_data_handler._read_json_or_jsonl(
            tmp_path / "statistics.json"
        )
    )

    assert len(train_set) == len(ids) == num_samples
    assert all(sum(value.values()) == num_samples for value in statistics.values())
