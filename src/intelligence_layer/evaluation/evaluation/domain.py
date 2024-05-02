import traceback
from datetime import datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, SerializeAsAny
from rich.tree import Tree

from intelligence_layer.evaluation.run.domain import RunOverview

Evaluation = TypeVar("Evaluation", bound=BaseModel, covariant=True)
"""Custom type that holds the domain-specific data of a single :class:`Example` evaluation"""


class FailedExampleEvaluation(BaseModel):
    """Captures an exception raised when evaluating an :class:`ExampleOutput`.

    Attributes:
        error_message: String-representation of the exception.
    """

    error_message: str

    @staticmethod
    def from_exception(exception: Exception) -> "FailedExampleEvaluation":
        return FailedExampleEvaluation(
            error_message=f"{type(exception)}: {str(exception)}\n{traceback.format_exc()}"
        )


class ExampleEvaluation(BaseModel, Generic[Evaluation]):
    """Evaluation of a single evaluated :class:`Example`

    Created to persist the evaluation result in the repository.

    Attributes:
        evaluation_id: Identifier of the run the evaluated example belongs to.
        example_id: Identifier of the :class:`Example` evaluated.
        result: If the evaluation was successful, evaluation's result,
            otherwise the exception raised during running or evaluating
            the :class:`Task`.

    Generics:
        Evaluation: Interface of the metrics that come from the evaluated :class:`Task`.
    """

    evaluation_id: str
    example_id: str
    result: SerializeAsAny[Evaluation | FailedExampleEvaluation]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"Evaluation ID = {self.evaluation_id}\n"
            f"Example ID = {self.example_id}\n"
            f"Result = {self.result}\n"
        )

    def _rich_render(self, skip_example_id: bool = False) -> Tree:
        tree = Tree(f"Evaluation: {self.evaluation_id}")
        if not skip_example_id:
            tree.add(f"Example ID: {self.example_id}")
        tree.add(str(self.result))
        return tree


class EvaluationOverview(BaseModel, frozen=True):
    """Overview of the un-aggregated results of evaluating a :class:`Task` on a dataset.

    Attributes:
        run_overviews: Overviews of the runs that were evaluated.
        id: The unique identifier of this evaluation.
        start: The time when the evaluation run was started
        description: human-readable for the evaluator that created the evaluation
    """

    run_overviews: frozenset[RunOverview]
    id: str
    start: Optional[datetime]
    description: str
    end_date: datetime
    successful_examples: int
    crashed_examples: int

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        run_overview_str: str = "Run Overviews={\n"
        comma_counter = 0
        for overview in self.run_overviews:
            run_overview_str += f"{overview}"
            if comma_counter < len(self.run_overviews) - 1:
                run_overview_str += ", "
                comma_counter += 1
        run_overview_str += "}\n"

        return (
            f"Evaluation Overview ID = {self.id}\n"
            f"Start time = {self.start}\n"
            f'Description = "{self.description}"\n'
            f"{run_overview_str}"
        )


class EvaluationFailed(Exception):
    def __init__(self, evaluation_id: str, failed_count: int) -> None:
        super().__init__(
            f"Evaluation {evaluation_id} failed with {failed_count} failed examples."
        )

    end: datetime
    failed_example_count: int
    successful_example_count: int
    description: str
