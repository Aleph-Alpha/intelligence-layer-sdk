from typing import Mapping

from pydantic import BaseModel

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language, language_config
from intelligence_layer.core.model import (
    AlephAlphaModel,
    CompleteInput,
    LuminousControlModel,
)
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import NoOpTracer, TaskSpan

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
    chunk: Chunk
    language: Language


class KeywordExtractOutput(BaseModel):
    keywords: frozenset[str]


class KeywordExtract(Task[KeywordExtractInput, KeywordExtractOutput]):
    def __init__(
        self,
        model: AlephAlphaModel = LuminousControlModel("luminous-base-control-20240215"),
        instruct_configs: Mapping[Language, str] = INSTRUCT_CONFIGS,
        maximum_tokens: int = 32,
    ) -> None:
        self._instruct_configs = instruct_configs
        self._model = model
        self._maximum_tokens = maximum_tokens

    def do_run(
        self, input: KeywordExtractInput, task_span: TaskSpan
    ) -> KeywordExtractOutput:
        instruction = language_config(input.language, self._instruct_configs)
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
