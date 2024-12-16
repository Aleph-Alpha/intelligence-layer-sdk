"""This code was taken from https://github.com/wandb/weave.

It follows the Apache License Version 2.0 January 2004 (http://www.apache.org/licenses/LICENSE-2.0)
Modified on 16.12.2024
"""

import ast
import inspect
import textwrap

from intelligence_layer.evaluation.aggregation.aggregator import AggregationLogic
from intelligence_layer.evaluation.evaluation.evaluator.evaluator import EvaluationLogic


class NotInteractiveEnvironmentError(Exception): ...


def is_running_interactively() -> bool:
    """Check if the code is running in an interactive environment."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ModuleNotFoundError:
        return False


def get_notebook_source() -> str:
    """Get the source code of the running notebook."""
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:
        raise NotInteractiveEnvironmentError

    if not hasattr(shell, "user_ns"):
        raise AttributeError("Cannot access user namespace")

    # This is the list of input cells in the notebook
    in_list = shell.user_ns["In"]

    # Stitch them back into a single "file"
    full_source = "\n\n".join(cell for cell in in_list[1:] if cell)

    return full_source


def get_class_source(cls: type) -> str:
    """Get the latest source definition of a class in the notebook."""
    notebook_source = get_notebook_source()
    tree = ast.parse(notebook_source)
    class_name = cls.__name__

    # We need to walk the entire tree and get the last one since that's the most version of the cls
    segment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            segment = ast.get_source_segment(notebook_source, node)

    if segment is not None:
        return segment

    raise ValueError(f"Class '{class_name}' not found in the notebook")


def get_source_notebook_safe(logic: EvaluationLogic | AggregationLogic) -> str:
    # In ipython, we can't use inspect.getsource on classes defined in the notebook
    logic_class = type(logic)
    try:
        src = inspect.getsource(logic_class)
    except OSError:
        if is_running_interactively() and inspect.isclass(logic_class):
            src = get_class_source(logic_class)
    return textwrap.dedent(src)
