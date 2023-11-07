from typing import Mapping
from aleph_alpha_client import Client

from intelligence_layer.core.complete import (
    FewShot,
    FewShotConfig,
    FewShotInput,
    PromptOutput,
)
from intelligence_layer.core.detect_language import (
    Language,
)
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.task import Task
from intelligence_layer.use_cases.summarize.summarize import (
    SingleChunkSummarizeInput,
    SummarizeOutput,
)


class SingleChunkFewShotSummarize(Task[SingleChunkSummarizeInput, SummarizeOutput]):
    """Summarises a section into a text of medium length.

    Generate a summary given a few-shot setup.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        few_shot_configs: A mapping of valid `Language` to `FewShotConfig` for each
            supported language.
        model: A valid Aleph Alpha model name.
        maximum_tokens: The maximum number of tokens to be generated.
    """

    def __init__(
        self,
        client: Client,
        few_shot_configs: Mapping[Language, FewShotConfig],
        model: str,
        maximum_tokens: int,
    ) -> None:
        self._few_shot_configs = few_shot_configs
        self._few_shot = FewShot(client)
        self._model = model
        self._maximum_tokens = maximum_tokens

    def run(
        self, input: SingleChunkSummarizeInput, logger: DebugLogger
    ) -> SummarizeOutput:
        prompt_output = self._get_prompt_and_complete(input, logger)
        return SummarizeOutput(summary=prompt_output.response.strip())

    def _get_prompt_and_complete(
        self, input: SingleChunkSummarizeInput, logger: DebugLogger
    ) -> PromptOutput:
        prompt_config = self._few_shot_configs.get(input.language)
        if not prompt_config:
            raise ValueError(f"Could not find `prompt_config` for {input.language}.")
        return self._few_shot.run(
            FewShotInput(
                few_shot_config=prompt_config,
                input=input.chunk,
                model=self._model,
                maximum_response_tokens=self._maximum_tokens,
            ),
            logger,
        )
