from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from sys import argv
from typing import Any, Sequence

from dotenv import load_dotenv

from intelligence_layer.connectors.limited_concurrency_client import (
    LimitedConcurrencyClient,
)
from intelligence_layer.evaluation.data_storage.aggregation_repository import (
    FileAggregationRepository,
)
from intelligence_layer.evaluation.data_storage.dataset_repository import (
    FileDatasetRepository,
)
from intelligence_layer.evaluation.data_storage.evaluation_repository import (
    FileEvaluationRepository,
)
from intelligence_layer.evaluation.data_storage.run_repository import FileRunRepository
from intelligence_layer.evaluation.runner import Runner


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
    parser.add_argument(
        "--description",
        required=True,
        type=str,
        help="Description of the evaluator.",
    )
    args = parser.parse_args(cli_args[1:])
    return args


def main(cli_args: Sequence[str]) -> None:
    args = parse_args(cli_args)
    dataset_repository = FileDatasetRepository(args.dataset_repository_path)
    runner_repository = FileRunRepository(args.target_dir)
    evaluation_repository = FileEvaluationRepository(args.target_dir)
    aggregation_repository = FileAggregationRepository(args.target_dir)
    description = args.description
    task = create_task(args.task)
    runner = Runner(task, dataset_repository, runner_repository, args.task.__name__)
    dataset_id = args.dataset_id
    run_overview_id = runner.run_dataset(dataset_id).id
    evaluator = args.evaluator(
        dataset_repository,
        runner_repository,
        evaluation_repository,
        aggregation_repository,
        description,
    )
    evaluator.eval_and_aggregate_runs(run_overview_id)


if __name__ == "__main__":
    load_dotenv()
    main(argv)
