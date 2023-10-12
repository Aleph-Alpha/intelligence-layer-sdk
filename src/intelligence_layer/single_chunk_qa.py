from typing import Optional, Sequence
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    Prompt,
)
from pydantic import BaseModel

from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from intelligence_layer.grading import (
    ExactMatchGrader,
    MockLlamaGrader,
    RandomListGrader,
)
from intelligence_layer.text_highlight import (
    TextHighlight,
    TextHighlightInput,
)
from intelligence_layer.prompt_template import (
    PromptTemplate,
    PromptWithMetadata,
)
from intelligence_layer.task import (
    DebugLogger,
    Evaluation,
    Evaluator,
    Task,
)


class SingleChunkQaInput(BaseModel):
    chunk: str
    question: str


class SingleChunkQaOutput(BaseModel):
    answer: Optional[str]
    highlights: Sequence[str]


class SingleChunkQa(Task[SingleChunkQaInput, SingleChunkQaOutput]):
    PROMPT_TEMPLATE_STR = """### Instruction:
{{question}}
If there's no answer, say "{{no_answer_text}}".

### Input:
{% promptrange text %}{{text}}{% endpromptrange %}

### Response:"""
    NO_ANSWER_STR = "NO_ANSWER_IN_TEXT"

    def __init__(
        self,
        client: Client,
        model: str = "luminous-supreme-control",
    ):
        self.client = client
        self.completion = Completion(client)
        self.text_highlight = TextHighlight(client)
        self.model = model

    def run(
        self, input: SingleChunkQaInput, logger: DebugLogger
    ) -> SingleChunkQaOutput:
        prompt_with_metadata = self._to_prompt_with_metadata(
            input.chunk, input.question
        )
        output = self._complete(
            prompt_with_metadata.prompt, logger.child_logger("Generate Answer")
        )
        highlights = self._get_highlights(
            prompt_with_metadata,
            output.completion(),
            logger.child_logger("Explain Answer"),
        )
        return SingleChunkQaOutput(
            answer=self._no_answer_to_none(output.completion().strip()),
            highlights=highlights,
        )

    def _to_prompt_with_metadata(self, text: str, question: str) -> PromptWithMetadata:
        template = PromptTemplate(self.PROMPT_TEMPLATE_STR)
        return template.to_prompt_with_metadata(
            text=text, question=question, no_answer_text=self.NO_ANSWER_STR
        )

    def _complete(self, prompt: Prompt, logger: DebugLogger) -> CompletionOutput:
        request = CompletionRequest(prompt)
        output = self.completion.run(
            CompletionInput(request=request, model=self.model), logger
        )
        return output

    def _get_highlights(
        self,
        prompt_with_metadata: PromptWithMetadata,
        completion: str,
        logger: DebugLogger,
    ) -> Sequence[str]:
        highlight_input = TextHighlightInput(
            prompt_with_metadata=prompt_with_metadata,
            target=completion,
            model=self.model,
        )
        highlight_output = self.text_highlight.run(highlight_input, logger)
        return [h.text for h in highlight_output.highlights if h.score > 0]

    def _no_answer_to_none(self, completion: str) -> Optional[str]:
        return completion if completion != self.NO_ANSWER_STR else None


class QaEvaluator(Evaluator[SingleChunkQaInput, Optional[str]]):
    """
    First version of what we imagine an evaluator to look for a given task.
    All current metrics delivered by the graders are mock metrics.
    """

    def __init__(self, client: Client, task: SingleChunkQa):
        self.task = task
        self.exact_match_grader = ExactMatchGrader()
        self.random_grader = RandomListGrader()
        self.llama_grader = MockLlamaGrader(client)

    def evaluate(
        self,
        input: SingleChunkQaInput,
        logger: DebugLogger,
        expected_output: Optional[str] = None,
    ) -> Evaluation:
        qa_output = self.task.run(input, logger)
        actual_output = qa_output.answer
        exact_match_result = self.exact_match_grader.grade(
            actual=actual_output, expected=expected_output
        )
        random_result = self.random_grader.grade(
            actual=actual_output,
            expected_list=[expected_output] if expected_output else [],
        )
        llama_result = self.llama_grader.grade(
            instruction=input.question,
            input=input.chunk,
            actual=actual_output,
            expected=expected_output,
        )
        return Evaluation(
            {
                "exact_match": exact_match_result,
                "random": random_result,
                "llama": llama_result,
            }
        )
