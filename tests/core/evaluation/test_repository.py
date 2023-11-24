
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from pytest import fixture

from intelligence_layer.core import FileEvaluationRepository, ExampleResult, TaskSpanTrace, EvaluationException

class DummyEvaluation(BaseModel):
    result: str

@fixture
def file_evaluation_repository(tmp_path: Path) -> FileEvaluationRepository:
    return FileEvaluationRepository(tmp_path)

@fixture
def task_span_trace() -> TaskSpanTrace:
    now = datetime.now()
    return TaskSpanTrace(traces=[], start=now, end=now, input="input", output="output")

def test_can_store_traces_in_file(file_evaluation_repository: FileEvaluationRepository, task_span_trace: TaskSpanTrace) -> None:
    run_id = "id"
    example_result = ExampleResult(example_id="example_id", result=DummyEvaluation(result="result"), trace=task_span_trace)
    
    file_evaluation_repository.store_example_result(run_id, example_result)

    assert file_evaluation_repository.evaluation_example_result(run_id, example_result.example_id, DummyEvaluation) == example_result

class DummyEvaluationWithExceptionStructure(BaseModel):
    error_message: str

def test_storing_exception_with_same_structure_as_type_still_deserializes_exception(file_evaluation_repository: FileEvaluationRepository, task_span_trace: TaskSpanTrace) -> None:
    exception: ExampleResult[DummyEvaluation] = ExampleResult(example_id="id", result=EvaluationException(error_message="error"), trace=task_span_trace)
    run_id = "id"

    file_evaluation_repository.store_example_result(run_id, exception)

    assert file_evaluation_repository.evaluation_example_result(run_id, exception.example_id, DummyEvaluationWithExceptionStructure) == exception 
   