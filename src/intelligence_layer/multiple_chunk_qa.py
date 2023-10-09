from intelligence_layer.single_chunk_qa import SingleDocumentQaInput, QaOutput, SingleChunkQa
from intelligence_layer.task import DebugLog, LogLevel, Task
from intelligence_layer.avalible_models import ControlModels
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    ExplanationRequest,
    ExplanationResponse,
    Prompt,
    TextScore,
)
from typing import List
from pydantic import BaseModel

class MultipleDocumentQaInput(BaseModel):
    documents = List[SingleDocumentQaInput]


class MultipleDocumentQa(Task[SingleDocumentQaInput, QaOutput]):
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

        
        self.single_document_qa = SingleChunkQa(client, log_level, model)
    

    def run(self, input: MultipleDocumentQaInput) -> QaOutput:
        """Executes the process for this use-case."""
        
        qa_outputs: List[QaOutput] = [self.single_document_qa.run(doc) for doc in input.documents]

        answers: List[str] = [output.answer for output in qa_outputs]

        completion = Completion(client, log_level)
        



