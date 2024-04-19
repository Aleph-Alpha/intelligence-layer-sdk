from enum import Enum
from typing import Mapping, Sequence

from pydantic import BaseModel

from intelligence_layer.core import (
    CompleteInput,
    ControlModel,
    Language,
    LuminousControlModel,
    Task,
    TaskSpan,
    TextChunk,
)


class UnsafeOutputFlag(Enum):
    HATESPEECH = "hatespeech"
    SEXUAL = "sexual"
    RELIGIOUS = "religious"
    MEDICAL = "medical"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITIES = "illegal_activities"
    VIOLENCE = "violence"
    INSULT = "insult"
    PROFANITY = "profanity"


ALL_FLAGS = "\n-".join([e.value for e in UnsafeOutputFlag])

INSTRUCTION_CONFIGS = {
    Language(
        "en"
    ): f"""Your task is to act as a safety classifier for text. There are the following categories:
-{ALL_FLAGS}

Your response should be a single line containing the labels of which categories the text fulfills.
Return the label 'safe' if the text is safe.
Please make sure your response is a single line.
""",
    Language(
        "de"
    ): f"""Deine Aufgabe besteht darin, als Sicherheitsklassifizierer für Text zu fungieren. Es gibt folgende Kategorien:
-{ALL_FLAGS}

Deine Antwort solle eine Zeile mit Labels der jeweiligen Kategorien umfassen, die der Text erfüllt.
Gib das Label 'safe' zurück, wenn der Text sicher ist.
Bitte stelle sicher, dass deine Antwort eine Zeile lang ist.
""",
}


class SafetyClassifyInput(BaseModel):
    text: TextChunk
    language: Language = Language("en")


class SafetyClassifyOutput(BaseModel):
    flags: Sequence[UnsafeOutputFlag]  # "safe" is an empty list.
    # Flag is only included if the task determines it should be flagged


class SafetyClassifier(Task[SafetyClassifyInput, SafetyClassifyOutput]):
    def __init__(
        self,
        model: ControlModel | None = None,
        instruction_configs: Mapping[Language, str] = INSTRUCTION_CONFIGS,
    ) -> None:
        self._model = model or LuminousControlModel("luminous-nextgen-7b-control-384k")
        self._instruction_configs = instruction_configs

    def do_run(
        self, input: SafetyClassifyInput, task_span: TaskSpan
    ) -> SafetyClassifyOutput:
        instruction = self._instruction_configs.get(input.language)

        if not instruction:
            raise ValueError(f"Could not find `prompt_config` for {input.language}.")
        completion = self._model.complete(
            CompleteInput(
                prompt=self._model.to_instruct_prompt(instruction, input.text),
            ),
            task_span,
        )
        completion_str = str(completion.completions[0].completion)
        potential_flags = completion_str.split(", ")
        flags = [
            UnsafeOutputFlag(flag.strip())
            for flag in potential_flags
            if hasattr(UnsafeOutputFlag, flag.strip().upper())
        ]
        return SafetyClassifyOutput(flags=flags)
