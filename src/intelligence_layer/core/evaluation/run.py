from importlib import import_module
from pathlib import Path
from sys import argv
from typing import Any, Sequence

from dotenv import load_dotenv

from intelligence_layer.core import FileEvaluationRepository


def function_from_string(fully_qualified_function_name: str) -> Any:
    mod_name, func_name = fully_qualified_function_name.rsplit(".", 1)
    mod = import_module(mod_name)
    return getattr(mod, func_name)


def main(args: Sequence[str]) -> None:
    (
        _,
        evaluator_function,
        task_function,
        dataset_function,
        evaluation_repository_directory,
    ) = args
    repository = FileEvaluationRepository(Path(evaluation_repository_directory))
    task = function_from_string(task_function)()
    evaluator = function_from_string(evaluator_function)(task, repository)
    dataset = function_from_string(dataset_function)()
    evaluator.evaluate_dataset(dataset)


if __name__ == "__main__":
    load_dotenv()
    main(argv)
