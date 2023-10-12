from abc import abstractmethod
import random
from typing import Any, Optional, Sequence

from aleph_alpha_client import Client

from intelligence_layer.completion import Completion


class Grader:
    ...


class GoldenAnswerGrader(Grader):
    @abstractmethod
    def grade(self, actual: Optional[str], expected: Optional[str]) -> Any:
        pass


class ListGrader(Grader):
    @abstractmethod
    def grade(
        self, actual: Optional[str], expected_list: Optional[Sequence[str]]
    ) -> Any:
        pass


class InstructionGrader(Grader):
    @abstractmethod
    def grade(
        self,
        instruction: str,
        input: Optional[str],
        actual: Optional[str],
        expected: Optional[str],
    ) -> Any:
        pass


class ExactMatchGrader(GoldenAnswerGrader):
    def grade(self, actual: Optional[str], expected: Optional[str]) -> bool:
        return actual == expected


class RandomListGrader(ListGrader):
    def grade(
        self, actual: Optional[str], expected_list: Optional[Sequence[str]]
    ) -> float:
        return random.random()


class MockLlamaGrader(InstructionGrader):
    def __init__(self, client: Client):
        self.client = client
        self.completion = Completion(client)

    def grade(
        self,
        instruction: str,
        input: Optional[str],
        actual: Optional[str],
        expected: Optional[str],
    ) -> str:
        return random.choice(["GREAT", "OK", "BAD"])
