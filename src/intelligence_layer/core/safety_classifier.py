from enum import Enum
from typing import Mapping, Sequence

from pydantic import BaseModel

from intelligence_layer.core import (
    Language,
    LuminousControlModel,
    Task,
    TaskSpan,
    TextChunk,
)
from intelligence_layer.use_cases import ClassifyInput, PromptBasedClassify

INSTRUCTION_CONFIGS = {
    Language(
        "en"
    ): """Your task is to act as a safety classifier for text. Identify a class that describes the text adequately.
    Reply with only the class label.""",
    Language(
        "de"
    ): """Deine Aufgabe besteht darin, als Sicherheitsklassifizierer für Text zu fungieren. Identifiziere eine Klasse,
    die den Text adäquat beschreibt. Antworte nur mit dem Label der Klasse.
    """,
}


class SafetyOutputFlag(Enum):
    SAFE = "safe"
    HATESPEECH = "hatespeech"
    SEXUAL = "sexual"
    RELIGIOUS = "religious"
    MEDICAL = "medical"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITIES = "illegal_activities"
    VIOLENCE = "violence"
    INSULT = "insult"


class SafetyClassifyInput(BaseModel):
    text: TextChunk
    language: Language = Language("en")


class SafetyClassifyOutput(BaseModel):
    flags: Sequence[SafetyOutputFlag]  # "safe" is an empty list.
    # Flag is only included if the task determines it should be flagged


class SafetyClassifier(Task[SafetyClassifyInput, SafetyClassifyOutput]):
    def __init__(
        self,
        model: LuminousControlModel | None = None,
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
        classify_inputs = ClassifyInput(
            chunk=input.text,
            labels=frozenset({flag.value for flag in SafetyOutputFlag}),
        )
        prompt_based_classify = PromptBasedClassify(
            model=self._model, instruction=instruction
        )
        output_probabilities_per_flag = prompt_based_classify.run(
            classify_inputs, task_span
        )

        most_probable_flag = SafetyOutputFlag(
            output_probabilities_per_flag.sorted_scores[0][0]
        )

        if most_probable_flag == SafetyOutputFlag.SAFE:
            return SafetyClassifyOutput(flags=[])

        return SafetyClassifyOutput(flags=[most_probable_flag])
