import math
from typing import Set, Mapping, Tuple, Optional, Sequence

from aleph_alpha_client import (
    Client,
    PromptTemplate,
    Tokens,
    Prompt,
    TokenizationRequest,
    CompletionRequest,
)
from pydantic import BaseModel

from ._task import Task, DebugLog


class ClassifyInput(BaseModel):
    """Input for a classification task."""

    text: str
    """Text to be classified"""
    labels: Set[str]
    """Possible labels into which the text should be classified."""


class ClassifyOutput(BaseModel):
    scores: Mapping[str, float]
    """Mapping of the provided label (key) to corresponding score (value)

    The score represents how sure the model is that this is the correct label.
    Will be a value between 0 and 1"""
    debug_log: DebugLog
    """Provides key steps, decisions, and intermediate outputs of a task's process."""


class SingleLabelClassify(Task[ClassifyInput, ClassifyOutput]):
    """Classify method for applying a single label to a given text.

    The input provides a complete set of all possible labels. The output will return a score for
    each possible label. All scores will add up to 1 and are relative to each other. The highest
    score is given to the most likely task.

    This methodology works best for classes that are easily understood, and don't require an
    explanation or examples."""

    PROMPT_TEMPLATE: PromptTemplate = PromptTemplate(
        """### Instruction:
Identify a class that describes the text adequately.
Reply with only the class label.

### Input:
{{text}}

### Response:{{label}}"""
    )
    MODEL: str = "luminous-base-control"
    client: Client

    def __init__(self, client: Client) -> None:
        """Initializes the Task.

        Args:
        - client: the aleph alpha client
        """
        super().__init__()
        self.client = client

    def run(self, input: ClassifyInput) -> ClassifyOutput:
        log = DebugLog()
        scores = self._calculate_scores(input, log)
        return ClassifyOutput(
            scores=scores,
            debug_log=log,
        )

    def _calculate_scores(
        self, input: ClassifyInput, log: DebugLog
    ) -> Mapping[str, float]:
        """Generates log probs for each label and generates a relative score for each"""
        tree = TreeNode()
        prompts = self._generate_prompts(input)
        log.add("The labels", [prompt[0] for prompt in prompts])
        for label, prompt, tokens in prompts:
            tree.insert_path(self._generate_log_probs(prompt, tokens=tokens))
        tree.normalize_probs()
        scores = tree.calculate_path_prob()

        return {
            # Make sure we are actually matching the correct label
            label: next(
                score for token_list, score in scores if token_list == tokens.tokens
            )
            for label, _, tokens in prompts
        }

    def _generate_log_probs(
        self, prompt: Prompt, tokens: Tokens
    ) -> Sequence[Tuple[int, float]]:
        """Generates a completion request that returns log probs via echo"""
        completion = self.client.complete(
            CompletionRequest(
                prompt=prompt, maximum_tokens=0, log_probs=0, tokens=True, echo=True
            ),
            self.MODEL,
        ).completions[0]

        assert isinstance(completion.log_probs, list)
        assert isinstance(completion.completion_tokens, list)
        assert len(completion.log_probs) == len(completion.completion_tokens)

        return [
            (token_id, log_probs[token] or 0.0)
            for token_id, log_probs, token in zip(
                tokens.tokens,
                completion.log_probs[-len(tokens.tokens) :],
                completion.completion_tokens[-len(tokens.tokens) :],
            )
            if log_probs[token] is not None
        ]

    def _generate_prompts(
        self, input: ClassifyInput
    ) -> Sequence[Tuple[str, Prompt, Tokens]]:
        """Embeds each label in a prompt. Label is tokenized separately so we know how many tokens
        to look at in the log probs"""
        tokenized_labels = (
            (label, self._tokenize_label(label)) for label in input.labels
        )
        return [
            (
                label,
                self.PROMPT_TEMPLATE.to_prompt(
                    text=input.text,
                    label=self.PROMPT_TEMPLATE.placeholder(tokens),
                ),
                tokens,
            )
            for (label, tokens) in tokenized_labels
        ]

    def _tokenize_label(self, label: str) -> Tokens:
        """Turn a single label into list of token ids. Important so that we know how many tokens
        the label is and can retrieve the last N log probs for the label"""
        response = self.client.tokenize(
            request=TokenizationRequest(
                label + "<|endoftext|>", tokens=False, token_ids=True
            ),
            model=self.MODEL,
        )
        assert isinstance(response.token_ids, list)
        return Tokens.from_token_ids(response.token_ids)


class TreeNode:
    def __init__(self, value: Optional[int] = None, log_prob: Optional[float] = None):
        self.value = value
        self.log_prob = log_prob
        self.normalized_prob: Optional[float] = None
        self.children: list[TreeNode] = []

    def find_child(self, value: int) -> Optional["TreeNode"]:
        for child in self.children:
            if child.value == value:
                return child
        return None

    def insert_path(self, path: Sequence[tuple[int, float]]) -> None:
        if not path:
            return
        value, log_prob = path[0]
        log_prob = math.exp(log_prob)

        child = self.find_child(value)
        if child is None:
            child = TreeNode(value, log_prob)
            self.children.append(child)

        child.insert_path(path[1:])

    def normalize_probs(self) -> None:
        total_prob = sum(
            child.log_prob for child in self.children if child.log_prob is not None
        )
        for child in self.children:
            if child.log_prob is not None:
                child.normalized_prob = child.log_prob / total_prob
            child.normalize_probs()

    def calculate_path_prob(
        self, normalized_prob: float = 1.0
    ) -> Sequence[tuple[list[int], float]]:
        path_probs = []

        for child in self.children:
            if child.normalized_prob is not None and child.log_prob is not None:
                new_normalized_prob = normalized_prob * child.normalized_prob

                child_paths = child.calculate_path_prob(new_normalized_prob)
                for path, normal_prob in child_paths:
                    path_probs.append(
                        (
                            [self.value] + path if self.value is not None else path,
                            normal_prob,
                        )
                    )

        if not path_probs and self.value is not None and self.normalized_prob:
            return [([self.value], normalized_prob)]

        return path_probs
