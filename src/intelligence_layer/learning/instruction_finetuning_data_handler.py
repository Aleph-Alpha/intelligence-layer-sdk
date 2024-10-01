import json
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from ctypes import c_int
from datetime import datetime
from enum import Enum
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Any, Optional, TypeVar

from aleph_alpha_client import Text
from sqlalchemy import ColumnElement

from intelligence_layer.core.detect_language import (
    DetectLanguage,
    DetectLanguageInput,
    DetectLanguageOutput,
    Language,
)
from intelligence_layer.core.model import (
    AlephAlphaChatModel,
    FinetuningMessage,
    Message,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer.tracer import NoOpTracer, Tracer
from intelligence_layer.learning.enrich import (
    EnrichDomain,
    EnrichmentInput,
    EnrichQuality,
)
from intelligence_layer.learning.instruction_finetuning_data_repository import (
    InstructionFinetuningDataRepository,
)
from intelligence_layer.learning.models import (
    InstructionFinetuningSample,
    InstructionFinetuningSampleAttributes,
    InvalidSampleError,
    RawInstructionFinetuningSample,
    TrainSet,
    TripletTransformation,
)

T = TypeVar("T")


class EnrichAction(Enum):
    REMOVE = "remove"
    SKIP = "skip"
    GET = "get"
    OVERWRITE = "overwrite"


class InstructionFinetuningDataHandler:
    """Acts as the interface between a user and his fine-tuning data.

    Takes a repository and tasks that carry "enrichment" or annotation tasks. Added
    data will be stored in repository and data will be annotated, e.g. with a `quality`-
    label.

    Args:
        repository: `InstructionFinetuningDataRepository` implementation. Responsible for
            efficient data storage, indexing & retrieval.
        domain_task: Task-implementation used for annotating domains.
        quality_task: Task-implementation used for annotating sample quality.
        language_task: Task-implementation used for detecting languages used.
        supported_languages: Languages that should be considered for detection & usage.
        default_language: Fallback language in case no languages are found.
        tracer: Tracer used for logging "enrichment"-activity.
    """

    def __init__(
        self,
        repository: InstructionFinetuningDataRepository,
        domain_task: Task[EnrichmentInput, Optional[str]],
        quality_task: Task[EnrichmentInput, Optional[int]],
        language_task: Task[DetectLanguageInput, DetectLanguageOutput],
        supported_languages: Sequence[Language],
        default_language: Language,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.repository = repository
        self.domain_task = domain_task
        self.quality_task = quality_task
        self.language_task = language_task
        self.supported_languages = supported_languages
        self.default_language = default_language
        self.tracer = tracer or NoOpTracer()

    def add_sample(
        self,
        raw_sample: RawInstructionFinetuningSample,
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
    ) -> str:
        validated_and_enriched_sample = self._validate_and_enrich_sample(
            raw_sample, domain_action, quality_action, language_action
        )
        return self.repository.store_sample(validated_and_enriched_sample)

    def add_samples(
        self,
        raw_samples: Iterable[RawInstructionFinetuningSample],
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
        max_workers: int = 10,
    ) -> list[str]:
        def _safe_validate_and_enrich_sample(
            raw_sample: RawInstructionFinetuningSample,
            domain_action: EnrichAction,
            quality_action: EnrichAction,
            language_action: EnrichAction,
        ) -> Optional[InstructionFinetuningSample]:
            try:
                return self._validate_and_enrich_sample(
                    raw_sample, domain_action, quality_action, language_action
                )
            except Exception as e:
                warnings.warn(
                    f"An error occurred for sample with external_id '{raw_sample.external_id}': {e!s}"
                )
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = (
                result
                for result in executor.map(
                    lambda raw_sample: _safe_validate_and_enrich_sample(
                        raw_sample, domain_action, quality_action, language_action
                    ),
                    raw_samples,
                )
                if result
            )

        return self.repository.store_samples(results)

    def update_sample(
        self,
        id: str,
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
    ) -> str:
        sample = self._update_sample(id, domain_action, quality_action, language_action)
        return self.repository.store_sample(sample)

    def update_samples(
        self,
        ids: Iterable[str],
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
        max_workers: int = 10,
    ) -> Iterable[str]:
        def _safe_update_sample(
            id: str,
            domain_action: EnrichAction,
            quality_action: EnrichAction,
            language_action: EnrichAction,
        ) -> Optional[InstructionFinetuningSample]:
            try:
                return self._update_sample(
                    id, domain_action, quality_action, language_action
                )

            except Exception as e:
                warnings.warn(f"An error occurred for sample with id '{id}': {e!s}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = (
                result
                for result in executor.map(
                    lambda id: _safe_update_sample(
                        id, domain_action, quality_action, language_action
                    ),
                    ids,
                )
                if result
            )

        yield from self.repository.store_samples(results)

    def head(self, limit: Optional[int] = 100) -> Iterable[InstructionFinetuningSample]:
        yield from self.repository.head(limit)

    def sample(self, id: str) -> Optional[InstructionFinetuningSample]:
        return self.repository.sample(id)

    def samples(self, ids: Iterable[str]) -> Iterable[InstructionFinetuningSample]:
        yield from self.repository.samples(ids)

    def samples_with_filter(
        self, filter_expression: ColumnElement[bool] | None, limit: int | None
    ) -> Iterable[InstructionFinetuningSample]:
        if filter_expression is not None:
            yield from self.repository.samples_with_filter(filter_expression, limit)
        else:
            yield from self.repository.head(limit)

    def delete_sample(self, id: str) -> None:
        self.repository.delete_sample(id)

    def delete_samples(self, ids: Iterable[str]) -> None:
        self.repository.delete_samples(ids)

    @staticmethod
    def samples_to_train_set(
        model: AlephAlphaChatModel,
        samples: Iterable[InstructionFinetuningSample],
        max_workers: int = 10,
    ) -> TrainSet:
        def process_sample(
            sample: InstructionFinetuningSample,
            # actual type is Synchronized[int] but declaring this will actually fail at runtime
            # only declaring Synchronized will trigger mypy
            emitted_counter: Synchronized,  # type: ignore
            statistics_counter: dict[Any, dict[Any, int]],
        ) -> Optional[tuple[Sequence[FinetuningMessage], str]]:
            prompt = model.to_chat_prompt(sample.messages)

            prompt_item = prompt.items[0]
            assert len(prompt.items) == 1 and isinstance(prompt_item, Text)

            tokenized = model.tokenize(prompt_item.text)
            if len(tokenized) <= model.context_size:
                for key, value in sample.attributes.model_dump().items():
                    statistics_counter[str(key)][str(value)] += 1
                return model.to_finetuning_sample(sample.messages), sample.id

            warnings.warn(
                f"Sample with id '{sample.id}' has {len(tokenized)} tokens, exceeding the supplied model's context size of {model.context_size}."
            )
            with emitted_counter.get_lock():
                emitted_counter.value += 1
            return None

        emitted_counter = Value(c_int, 0)
        statistics_counter: dict[Any, dict[Any, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                result
                for result in executor.map(
                    lambda sample: process_sample(
                        sample, emitted_counter, statistics_counter
                    ),
                    samples,
                )
                if result
            )

        warnings.warn(
            f"Emitted {emitted_counter.value} sample(s) due to context size constraints."
        )

        return TrainSet(
            data=[result[0] for result in results],
            ids=[result[1] for result in results],
            statistics=statistics_counter,
        )

    def _validate_and_enrich_sample(
        self,
        raw_sample: RawInstructionFinetuningSample,
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
    ) -> InstructionFinetuningSample:
        self._validate_sample(raw_sample)
        sample = InstructionFinetuningSample.from_raw_sample(raw_sample)
        return self._enrich_sample(
            sample, domain_action, quality_action, language_action
        )

    @staticmethod
    def _validate_sample(raw_sample: RawInstructionFinetuningSample) -> None:
        counter: dict[str, int] = defaultdict(int)
        last_role: Optional[str] = None

        for message in raw_sample.messages:
            if not message.content.strip():
                raise InvalidSampleError(
                    f"Sample with external_id '{raw_sample.external_id}' does not contain anything."
                )

            if last_role is None and message.role == "assistant":
                raise InvalidSampleError(
                    f"Sample with external_id '{raw_sample.external_id}' starts with an assistant message."
                )

            if message.role == last_role:
                raise InvalidSampleError(
                    f"Sample with external_id '{raw_sample.external_id}' has multiple {last_role} messages right after another."
                )

            counter[message.role] += 1

            last_role = message.role

        if sum(counter.values()) <= 1:
            raise InvalidSampleError(
                f"Sample with external_id '{raw_sample.external_id}' has less than 2 messages."
            )

        else:
            if counter["system"] > 1:
                raise InvalidSampleError(
                    f"Sample with external_id '{raw_sample.external_id}' has multiple system messages."
                )

            if counter["assistant"] == 0:
                raise InvalidSampleError(
                    f"Sample with external_id '{raw_sample.external_id}' has no assistant messages."
                )

            if last_role != "assistant":
                raise InvalidSampleError(
                    f"Sample with external_id '{raw_sample.external_id}' ends on a {last_role} message."
                )

    def _enrich_sample(
        self,
        sample: InstructionFinetuningSample,
        domain_action: EnrichAction,
        quality_action: EnrichAction,
        language_action: EnrichAction,
    ) -> InstructionFinetuningSample:
        languages = self._enrich_attribute(
            sample.attributes.languages,
            language_action,
            self._get_sample_language,
            sample.messages,
        )
        languages_list = list(languages) if languages else []
        main_language = next(iter(languages_list), None) or self.default_language

        return InstructionFinetuningSample(
            messages=sample.messages,
            attributes=InstructionFinetuningSampleAttributes(
                source=sample.attributes.source,
                domain=self._enrich_attribute(
                    sample.attributes.domain,
                    domain_action,
                    self._get_sample_domain,
                    sample.messages,
                    main_language,
                ),
                quality=self._enrich_attribute(
                    sample.attributes.quality,
                    quality_action,
                    self._get_sample_quality,
                    sample.messages,
                    main_language,
                ),
                languages=languages,
            ),
            external_id=sample.external_id,
            id=sample.id,
        )

    def _enrich_attribute(
        self,
        attr: Optional[T],
        action: EnrichAction,
        func: Callable[..., Optional[T]],
        *args: Any,
    ) -> Optional[T]:
        if action == EnrichAction.REMOVE:
            return None
        if action == EnrichAction.SKIP:
            return attr
        if action == EnrichAction.GET:
            return attr if attr else func(*args)
        if action == EnrichAction.OVERWRITE:
            return func(*args)

    def _get_sample_language(
        self, messages: Sequence[Message]
    ) -> Optional[Sequence[Language]]:
        outputs = [
            self.language_task.run(
                input=DetectLanguageInput(
                    text=m.content, possible_languages=self.supported_languages
                ),
                tracer=self.tracer,
            )
            for m in messages
        ]
        languages = [o.best_fit for o in outputs if o.best_fit]
        if languages:
            return list(set(languages))
        return None

    def _get_sample_domain(
        self, messages: Sequence[Message], language: Language
    ) -> Optional[str]:
        return self.domain_task.run(
            EnrichmentInput(messages=messages, language=language), self.tracer
        )

    def _get_sample_quality(
        self, messages: Sequence[Message], language: Language
    ) -> Optional[int]:
        return self.quality_task.run(
            EnrichmentInput(messages=messages, language=language), self.tracer
        )

    def _update_sample(
        self,
        id: str,
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
    ) -> InstructionFinetuningSample:
        sample_to_update = self.repository.sample(id)
        if sample_to_update:
            enriched_sample = self._enrich_sample(
                sample_to_update, domain_action, quality_action, language_action
            )
            return enriched_sample

        raise KeyError(f"No sample found for id '{id}'.")

    def add_chat_messages_data_from_file(
        self,
        path: Path,
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
        max_workers: int = 10,
    ) -> None:
        data = self._read_json_or_jsonl(path)
        samples = [
            InstructionFinetuningSample.from_chat_messages_json(datapoint, path.stem)
            for datapoint in data
        ]
        self.add_samples(
            samples, domain_action, quality_action, language_action, max_workers
        )

    def add_triplet_data_from_file(
        self,
        path: Path,
        domain_action: EnrichAction = EnrichAction.GET,
        quality_action: EnrichAction = EnrichAction.GET,
        language_action: EnrichAction = EnrichAction.GET,
        max_workers: int = 10,
        triplet_transformation: TripletTransformation = TripletTransformation.INSTRUCTION_AS_SYSTEM,
    ) -> None:
        data = self._read_json_or_jsonl(path)
        samples = [
            InstructionFinetuningSample.from_triplet_json(
                datapoint, path.stem, triplet_transformation
            )
            for datapoint in data
        ]
        self.add_samples(
            samples, domain_action, quality_action, language_action, max_workers
        )

    @staticmethod
    def _read_json_or_jsonl(path: Path) -> Any:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as file:
                return [json.loads(line) for line in file]

        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def save_jsonl(data: Sequence[Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def save_json(data: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def compile_train_set(
        self,
        save_dir: Path,
        model: AlephAlphaChatModel,
        filter_expression: ColumnElement[bool] | None = None,
        limit: int | None = None,
    ) -> Path:
        """Method to sample a finetuning train set based on some filter expressions.

        Args:
            save_dir: The path to save the dataset to. Will open new subdir within
                with date as directory name.
            model: Model specifies tokenizer for token counting & prompt format
            filter_expression: Optional filtering statement
            limit: Optional parameter to indicate maximum sample count

        Returns:
            The directory the train set was saved to.
        """
        samples = self.samples_with_filter(filter_expression, limit)
        train_set = self.samples_to_train_set(model, samples)

        save_path = save_dir / datetime.now().strftime("Y-%m-%d_%H-%M-%S")
        (
            self.save_jsonl(
                [
                    [message.model_dump() for message in datapoint]
                    for datapoint in train_set.data
                ],
                save_path / "train_set.jsonl",
            ),
        )
        self.save_jsonl(train_set.ids, save_path / "ids.jsonl")
        self.save_json(train_set.statistics, save_path / "statistics.json")
        return save_path


def instruction_finetuning_handler_builder(
    repository: InstructionFinetuningDataRepository, domains: list[str]
) -> InstructionFinetuningDataHandler:
    return InstructionFinetuningDataHandler(
        repository=repository,
        domain_task=EnrichDomain(domains),
        quality_task=EnrichQuality(),
        language_task=DetectLanguage(),
        supported_languages=[Language("en"), Language("de")],
        default_language=Language("en"),
    )
