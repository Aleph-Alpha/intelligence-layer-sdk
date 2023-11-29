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


def function_from_string(fully_qualified_function_name: str) -> Any:
    mod_name, func_name = fully_qualified_function_name.rsplit(".", 1)
    mod = import_module(mod_name)
    return getattr(mod, func_name)


def main(cli_args: Sequence[str]) -> None:
    args = parse_args(cli_args)
    repository = FileEvaluationRepository(args.target_dir)
    task = create_task(args.task)
    evaluator = args.evaluator(task, repository)
    dataset = args.dataset()
    evaluator.evaluate_dataset(dataset)


def create_task(factory: Any) -> Any:
    try:
        return factory()
    except TypeError:
        return factory(LimitedConcurrencyClient.from_token())


def parse_args(cli_args: Sequence[str]) -> Namespace:
    parser = ArgumentParser(description="Runs the given evaluation")
    parser.add_argument("--evaluator", required=True, type=function_from_string)
    parser.add_argument("--task", required=True, type=function_from_string)
    parser.add_argument("--dataset", required=True, type=function_from_string)
    parser.add_argument("--target-dir", required=True, type=Path)
    args = parser.parse_args(cli_args[1:])
    return args


if __name__ == "__main__":
    load_dotenv()
    main(argv)
