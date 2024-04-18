from enum import Enum
from typing import Sequence

from pydantic import BaseModel

from intelligence_layer.core import (
    CompleteInput,
    ControlModel,
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


class SafetyClassifyInput(BaseModel):
    text: TextChunk


class SafetyClassifyOutput(BaseModel):
    flags: Sequence[UnsafeOutputFlag]  # "safe" is an empty list.
    # Flag is only included if the task determines it should be flagged


class SafetyClassifier(Task[SafetyClassifyInput, SafetyClassifyOutput]):
    def __init__(
        self,
        model: ControlModel | None = None,
    ) -> None:
        self._model = model or LuminousControlModel("luminous-nextgen-7b-control-384k")

    def do_run(
        self, input: SafetyClassifyInput, task_span: TaskSpan
    ) -> SafetyClassifyOutput:
        instruction = """Your task is to act as a safety classifier for text. There are the following categories:
-hatespeech,
-sexual,
-religious,
-medical,
-self_harm,
-illegal_activities,
-violence,
-insult,
-profanity

Your response should be a single line containing the labels of which categories the text fulfills.
Return the label 'safe' if the text is safe.
Please make sure your response is a single line.
"""

        if not instruction:
            raise ValueError("Could not find `instruction` for safety classifier.")
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
