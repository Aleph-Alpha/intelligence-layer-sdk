from typing import Mapping

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

INSTRUCT_CONFIGS = {
    Language(
        "de"
    ): "Worum geht es in dem Text? Extrahiere ein paar Stichwörter in Form einer Komma-separierten Liste.",
    Language(
        "en"
    ): "What is the text about? Extract a few keywords in form of a comma-separated list.",
    Language(
        "es"
    ): "¿De qué trata el texto? Extrae algunas palabras clave en forma de una lista separada por comas.",
    Language(
        "fr"
    ): "De quoi parle le texte? Extraire quelques mots-clés sous forme d'une liste séparée par des virgules.",
    Language(
        "it"
    ): "Di cosa tratta il testo? Estrai alcune parole chiave sotto forma di una lista separata da virgole.",
}


class KeywordExtractInput(BaseModel):
    chunk: TextChunk
    language: Language


class KeywordExtractOutput(BaseModel):
    keywords: frozenset[str]


class KeywordExtract(Task[KeywordExtractInput, KeywordExtractOutput]):

    def __init__(
        self,
        model: ControlModel | None = None,
        instruct_configs: Mapping[Language, str] = INSTRUCT_CONFIGS,
        maximum_tokens: int = 32,
    ) -> None:
        self._instruct_configs = instruct_configs
        self._model = model or LuminousControlModel("luminous-base-control")
        self._maximum_tokens = maximum_tokens

    def do_run(
        self, input: KeywordExtractInput, task_span: TaskSpan
    ) -> KeywordExtractOutput:
        instruction = input.language.language_config(self._instruct_configs)
        result = self._model.complete(
            CompleteInput(
                prompt=self._model.to_instruct_prompt(
                    instruction=instruction, input=str(input.chunk)
                ),
                maximum_tokens=self._maximum_tokens,
            ),
            task_span,
        )
        return KeywordExtractOutput(
            keywords=frozenset(s.strip() for s in result.completion.split(","))
        )
