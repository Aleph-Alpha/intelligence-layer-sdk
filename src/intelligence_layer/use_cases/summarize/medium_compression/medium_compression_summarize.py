from typing import Sequence

from aleph_alpha_client import Client

from intelligence_layer.core.complete import (
    FewShot,
    FewShotInput,
    PromptOutput,
)
from intelligence_layer.core.detect_language import DetectLanguage, DetectLanguageInput
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.use_cases.summarize.base.base_summarize import (
    BaseSummarize,
    SummarizeInput,
)

from .prompt_configs import prompt_configs


class MediumCompressionSummarize(BaseSummarize):
    """Summarises a section into a text of medium length.

    Generate a short body natural language summary.
    Will also return highlights explaining which parts of the input contributed strongly to the completion.

    Note:
        `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        MAXIMUM_RESPONSE_TOKENS: The maximum number of tokens the summary will contain.
        INSTRUCTION: The verbal instruction sent to the model to make it generate the summary.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = InstructSummarize(client)
        >>> input = SummarizeInput(
        >>>     chunk="This is a story about pizza. Tina hates pizza. However, Mike likes it. Pete strongly believes that pizza is the best thing to exist."
        >>> )
        >>> logger = InMemoryLogger(name="Short Body Summarize")
        >>> output = task.run(input, logger)
        >>> print(output.summary)
        Tina does not like pizza, but Mike and Pete do.
    """

    PROMPT_CONFIGS = prompt_configs
    DEFAULT_LANG = "en"
    _client: Client

    def __init__(self, client: Client, model: str = "luminous-supreme-control") -> None:
        super().__init__(client, model)
        self._detect_language = DetectLanguage()
        self._few_shot = FewShot(client)

    def get_prompt_and_complete(
        self, input: SummarizeInput, logger: DebugLogger
    ) -> PromptOutput:
        language = self._get_language(input.chunk, logger)
        prompt_config = self.PROMPT_CONFIGS.get(language)
        if not prompt_config:
            raise ValueError(
                "Could not find `prompt_config` for the detected language."
            )
        return self._few_shot.run(
            FewShotInput(input=input.chunk, few_shot_config=prompt_config),
            logger,
        )

    def _get_language(self, text: str, logger: DebugLogger) -> str:
        detect_language_input = DetectLanguageInput(
            text=text, possible_languages=list(self.PROMPT_CONFIGS.keys())
        )
        detect_language_output = self._detect_language.run(
            detect_language_input, logger
        )
        return detect_language_output.best_fit or self.DEFAULT_LANG
