import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

from liquid import Template
from pydantic import BaseModel

from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.model import ChatModel, Llama3ChatModel, Message
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer.tracer import TaskSpan


class EnrichmentInput(BaseModel):
    """The input to every enrichment action.

    Args:
        messages: A number of messages making up one "chat" or "sample"
        language: Language to be used by enrichment task
    """

    messages: Sequence[Message]
    language: Language


class EnrichDomainConfig(BaseModel):
    """Prompt configuration for task that annotates the domain for each sample.

    Args:
        prompt_template: Instruction for the task
        system_verbose: Prefix for chat messages by the "system"
        user_verbose: Prefix for chat messages by the "user"
        assistant_verbose: Prefix for chat messages by the "assistant"
    """

    prompt_template: str
    system_verbose: str
    user_verbose: str
    assistant_verbose: str


ENRICH_DOMAIN_INSTRUCTIONS = {
    Language("en"): EnrichDomainConfig(
        prompt_template="Classify the given text into one of these domains: {{domains}}.",
        system_verbose="System instructions",
        user_verbose="User message",
        assistant_verbose="Assistant message",
    ),
    Language("de"): EnrichDomainConfig(
        prompt_template="Ordne den gegebenen Text in eine der folgenden Domänen ein: {{domains}}.",
        system_verbose="Systeminstruktionen",
        user_verbose="Nutzernachricht",
        assistant_verbose="Assistentennachricht",
    ),
}


class EnrichDomain(Task[EnrichmentInput, Optional[str]]):
    """Task that finds a matching domain for a sample.

    Args:
        domains: A list of domains that any given sample should be classified as
        chat_model: A multi-turn capable model to be used for domain generation
        instruction_config: Specifies prompt details to be used for requests
    """

    def __init__(
        self,
        domains: Sequence[str],
        chat_model: Optional[ChatModel] = None,
        instruction_config: Mapping[
            Language, EnrichDomainConfig
        ] = ENRICH_DOMAIN_INSTRUCTIONS,
    ) -> None:
        self._domains = domains
        self._chat_model = chat_model or Llama3ChatModel()
        self._instruction_config = instruction_config

    def do_run(self, input: EnrichmentInput, task_span: TaskSpan) -> Optional[str]:
        instruction_config = input.language.language_config(self._instruction_config)
        instruction = Template(instruction_config.prompt_template).render(
            domains=", ".join(self._domains)
        )
        generation = self._chat_model.generate_chat(
            messages=[
                Message(role="system", content=instruction),
                Message(
                    role="user",
                    content=self.input_messages_to_text(
                        input.messages, instruction_config
                    ),
                ),
            ],
            response_prefix=None,
            tracer=task_span,
        )
        return next((d for d in self._domains if d in generation), None)

    @staticmethod
    def input_messages_to_text(
        messages: Sequence[Message], instruction_config: EnrichDomainConfig
    ) -> str:
        role_mapping = {
            "system": instruction_config.system_verbose,
            "user": instruction_config.user_verbose,
            "assistant": instruction_config.assistant_verbose,
        }

        def get_role(role: Literal["system", "user", "assistant"]) -> str:
            if role not in role_mapping:
                raise ValueError(f"Got unexpected role in messages: {role}.")
            return role_mapping[role]

        return "\n\n".join(f"{get_role(m.role)}: {m.content}" for m in messages)


class EnrichQualityConfig(BaseModel):
    """Prompt configuration for task that annotates the domain for each sample.

    Args:
        system_prompt: System prompt to be used
        final_user_prompt: Final user prompt template used for "criticizing" the chat
        grading_scale: Maps natural language grade in prompt to integer value
    """

    system_prompt: str
    final_user_prompt: str
    grading_scale: Mapping[Any, int]


ENRICH_QUALITY_INSTRUCTIONS = {
    Language("en"): EnrichQualityConfig(
        system_prompt="You pretend to be an AI assistant assisting the user with his queries. At the end, you will be asked to critique your own responses with regard to their helpfulness.",
        final_user_prompt="""Now, critique all past responses.
The score should be given in the form of an American school grade, with "A" meaning exceptional performance and "F" meaning bad performance. Please respond with a JSON representing the evaluation. Respond in the format:
```
{
    "explanation": "One short and concise sentence explaining the evaluation, avoiding any potential bias. Use no more than 3 sentences.",
    "grade": Literal[A, B, C, D, E, F]
}
```
""",
        grading_scale={"A": 5, "B": 4, "C": 3, "D": 2, "F": 1},
    ),
    Language("de"): EnrichQualityConfig(
        system_prompt="Gib vor, ein KI-Assistent zu sein, der dem Benutzer bei seinen Fragen hilft. Am Ende wirst du gebeten, deine eigenen Antworten auf ihre Hilfsbereitschaft hin zu überprüfen.",
        final_user_prompt="""Bewerte nun alle bisherigen Antworten.
Die Bewertung sollte in Form einer deutschen Schulnote erfolgen, wobei "1" für hervorragende Leistung und "6" für schlechte Leistung steht. Bitte antworte mit einem JSON, das die Bewertung darstellt. Antworten in diesem Format:
```
{
    "explanation": „Ein kurzer und prägnanter Satz zur Erläuterung der Bewertung, wobei eine mögliche Voreingenommenheit zu vermeiden ist. Verwenden Sie nicht mehr als 3 Sätze.“,
    "grade": Literal[1, 2, 3, 4, 5, 6]
}
```
""",
        grading_scale={"1": 5, "2": 4, "3": 3, "4": 2, "5": 1, "6": 1},
    ),
}


class EnrichQuality(Task[EnrichmentInput, Optional[int]]):
    """Task that annotates the quality of a sample.

    Args:
        chat_model: A multi-turn capable model to be used for domain generation
        instruction_config: Specifies prompt details to be used for requests
    """

    def __init__(
        self,
        chat_model: Optional[ChatModel] = None,
        instruction_config: Mapping[
            Language, EnrichQualityConfig
        ] = ENRICH_QUALITY_INSTRUCTIONS,
    ) -> None:
        self._chat_model = chat_model or Llama3ChatModel(name="llama-3.1-70b-instruct")
        self._instruction_config = instruction_config

    def do_run(self, input: EnrichmentInput, task_span: TaskSpan) -> Optional[int]:
        instruction_config = input.language.language_config(self._instruction_config)
        response_prefix = "```\n{"
        generation = response_prefix + self._chat_model.generate_chat(
            messages=[
                Message(role="system", content=instruction_config.system_prompt),
                *input.messages,
                Message(role="user", content=instruction_config.final_user_prompt),
            ],
            response_prefix=response_prefix,
            tracer=task_span,
        )
        try:
            return self._parse_response(generation, instruction_config)
        except Exception as _:
            return None

    @staticmethod
    def _parse_response(
        generation: str, instruction_config: EnrichQualityConfig
    ) -> Optional[int]:
        loaded_json: Mapping[str, Any] = json.loads(generation.replace("```", ""))
        generated_grade = loaded_json.get("grade")
        return instruction_config.grading_scale.get(generated_grade)
