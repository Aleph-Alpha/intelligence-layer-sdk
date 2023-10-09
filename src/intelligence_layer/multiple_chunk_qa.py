from intelligence_layer.single_chunk_qa import SingleChunkQaInput, QaOutput, SingleChunkQa
from intelligence_layer.task import DebugLog, LogLevel, Task
from intelligence_layer.available_models import ControlModels
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    ExplanationRequest,
    ExplanationResponse,
    Prompt,
    TextScore,
)
from intelligence_layer.completion import Completion, CompletionInput, CompletionOutput
from typing import List
from pydantic import BaseModel

class MultipleChunkQaInput(BaseModel):
    chunks: List[str]
    question: str


class MultipleChunkQa(Task[SingleChunkQaInput, QaOutput]):
    PROMPT_TEMPLATE="""### Instruction:
You will be given a number of Answers to a Question. Based on them, generate a single final answer.
Condense multiple answers into a single answer. Rely only on the provided answers. Don't use the world's knowledge. The answer should combine the individual answers. If the answers contradict each other, e.g., one saying that the colour is green and the other saying that the colour is black, say that there are contradicting answers saying the colour is green or the colour is black.

### Input:
Question: {{question}}

Answers:
{{answers}}

### Response:
Final answer:"""

    def __init__(
        self,
        client: Client,
        log_level: LogLevel,
        model: ControlModels = ControlModels.SUPREME_CONTROL,
    ):

        self.client = client
        self.log_level = log_level
        self.completion = Completion(client, log_level)
        self.single_chunk_qa = SingleChunkQa(client, log_level, model)
    

    def run(self, input: MultipleChunkQaInput) -> QaOutput:
        """Executes the process for this use-case."""
        
        qa_outputs: List[QaOutput] = [self.single_chunk_qa.run(doc) for doc in input.documents]

        answers: List[str] = [output.answer for output in qa_outputs]

        return QaOutput(answer="XXX", highlights=["xxxx", "zzzz"], debug_log=None)



        
        



