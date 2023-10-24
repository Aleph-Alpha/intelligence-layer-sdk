from typing import NewType, Sequence
from aleph_alpha_client import Prompt
from pydantic import BaseModel

from intelligence_layer.task import DebugLogger, Task

LogProb = NewType("LogProb", float)
Probability = NewType("Probability", float)


class Token(BaseModel):
    """A token class containing it's id and the raw token.

    This is used instead of the Aleph Alpha client Token class since this one is serializable,
    while the one from the client is not.
    """

    token: str
    token_id: int


class TokenWithProb(BaseModel):
    token: Token
    prob: Probability | LogProb


class EchoInput(BaseModel):
    prompt: Prompt


class EchoOutput(BaseModel):
    tokens_with_log_probs: Sequence[Token]


class EchoTask(Task[EchoInput, EchoOutput]):
    def run(self, input: EchoInput, logger: DebugLogger) -> EchoOutput:
        return EchoOutput()