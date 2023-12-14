from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from sys import argv
from typing import Any, Sequence

from dotenv import load_dotenv

from intelligence_layer.connectors.limited_concurrency_client import (
    LimitedConcurrencyClient,
)
from intelligence_layer.core import FileEvaluationRepository
from intelligence_layer.core.evaluation.dataset_repository import FileDatasetRepository


def function_from_string(fully_qualified_function_name: str) -> Any:
    mod_name, func_name = fully_qualified_function_name.rsplit(".", 1)
    mod = import_module(mod_name)
    return getattr(mod, func_name)


def create_task(factory: Any) -> Any:
    try:
        return factory()
    except TypeError:
        return factory(LimitedConcurrencyClient.from_token())


def parse_args(cli_args: Sequence[str]) -> Namespace:
    parser = ArgumentParser(description="Runs the given evaluation")
    parser.add_argument(
        "--evaluator",
        required=True,
        type=function_from_string,
        help="A factory function for the evaluator. "
        "This function has to take 2 arguments: an instance of the Task to be evaluated and "
        "an EvaluationRepository where the results are stored. "
        "If this corresponds to the init-parameters of the Evaluator "
        "the class-type can actually be provided as argument.",
    )
    parser.add_argument(
        "--task",
        required=True,
        type=function_from_string,
        help="A factory function for the task to be evaluated. "
        "This function can either take no parameter or an Aleph Alpha client as only parameter. "
        "If this corresponds to the init-parameters of the Evaluator "
        "the class-type can actually be provided as argument.",
    )
    parser.add_argument(
        "--dataset-repository-path",
        required=True,
        type=Path,
        help="Path to a file dataset repository.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        type=str,
        help="ID of a dataset that exists in the file dataset repository provided.",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Path to a directory where the evaluation results are stored. "
        "The directory is created if it does not exist. "
        "The process must have corresponding write permissions.",
    )
    args = parser.parse_args(cli_args[1:])
    return args


def main(cli_args: Sequence[str]) -> None:
    args = parse_args(cli_args)
    evaluation_repository = FileEvaluationRepository(args.target_dir)
    dataset_repository = FileDatasetRepository(args.dataset_repository_path)
    task = create_task(args.task)
    evaluator = args.evaluator(task, evaluation_repository, dataset_repository)
    dataset_id = args.dataset_id
    evaluator.evaluate_dataset(dataset_id)


if __name__ == "__main__":
    load_dotenv()
    main(argv)
