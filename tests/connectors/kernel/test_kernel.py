from dotenv import load_dotenv
from pydantic import BaseModel

from intelligence_layer.connectors.kernel.kernel import KernelTask
from intelligence_layer.core.tracer.tracer import NoOpTracer


def test_kernel_connector() -> None:
    load_dotenv()
    tracer = NoOpTracer()

    class Input(BaseModel):
        question: str

    class Output(BaseModel):
        answer: str | None

    task = KernelTask(
        skill="app/super_rag",
        input_model=Input,
        output_model=Output,
    )

    output = task.run(
        Input(question="What is a transformer?"),
        tracer,
    )
    assert output.answer and "transformer" in output.answer
