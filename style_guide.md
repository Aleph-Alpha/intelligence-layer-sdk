# Intelligence Layer Style Guideline

Welcome to the project's style guide, a foundational document that ensures consistency, clarity, and quality in our collaborative efforts.
As we work together, adhering to the guidelines outlined here will streamline our process, making our code more readable and maintainable for all team members.

## Folder Structure
The source directory is organized into four distinct folders, each with a specific responsibility.

| **Folder**   | **Description**                                                                 |
|----------------|---------------------------------------------------------------------------------|
| Core           | The main components of the IL. This includes the `Task` abstraction, the `Tracer` and basic components like the `models`. |
| Evaluation     | Includes all resources related to task evaluation.                               |
| Connectors     | Provides tools to connect with third-party applications within the IL.    |
| Examples       | Showcases various task implementations to address different use cases using the IL.                  |

## Python: Naming Conventions

We follow the [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/).
In addition, there are the following naming conventions:
* Class method names:
  * Use only substantives for a method name having no side effects and returning some objects
    * E.g., `evaluation_overview` which returns an evaluation overview object
  * Use a verb for a method name if it has side effects and return nothing
    * E.g., `store_evaluation_overview` which saves a given evaluation overview (and returns nothing)


## Docstrings

### Task documentation

Document any `Task` like so:
``` python
class MyTask:
    """Start with a one-line description of the task, like this.

    Follow up with a more detailed description, outlining the purpose & general functioning of the task.

    Note:
        What is important? Does your task require a certain type of model? Any usage recommendations?

    Attributes:
        EXAMPLE_CONSTANT: Any constant that may be defined within the class.
        example_non_private_attribute: Any attribute defined within the '__init__' that is not private.

    Example:
        >>> var = "Describe here how to use this task end to end"
        >>> print("End on one newline.")
        End on one newline.
    """
```
The Example documentation is optional but preferred to be included in a how-to guide if it would be helpful in this case. 

Do not document the `run`` function of a class. Avoid documenting any other (private) functions.

### Input and output documentation

Document the inputs and outputs for a specific task like so:

``` python
class MyInput(BaseModel):
    """This is the input for this (suite of) task(s).

    Attributes:
        horse: Everybody knows what a horse is.
        chunk: We know what a chunk is, but does a user?
        crazy_deep_llm_example_param: Yeah, this probably deserves some explanation.
    """

# Any output should be documented in a similar manner
```

### Defaults

Certain parameters in each task are recurring. Where possible, we shall try to use certain standard documentation.

``` python
"""
client: Aleph Alpha client instance for running model related API calls.
model: A valid Aleph Alpha model name.
"""
```

### Module documentation

We **do not document the module**, as we assume imports like:

``` python
from intelligence_layer.complete import Complete
completion_task = Complete()
```

rather than:

``` python
from intelligence_layer import complete
completion_task = complete.Complete()
```

This ensures that the documentation is easily accessible by hovering over the imported task.

Generally, adhere to this [guideline](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

## Documentation Guide: Jupyter Notebooks vs. How-tos vs. Docstrings

When documenting our codebase, we focus on three primary channels: Jupyter notebooks, How-tos and docstrings.
The objective is to provide both a high-level understanding and detailed implementation specifics.
Here's how we differentiate and allocate content between them:

### Jupyter Notebooks

**Purpose**: Jupyter notebooks are used to provide a comprehensive overview and walkthrough of the tasks. They are ideal for understanding the purpose, usage, and evaluation of a task. (look [here](#when-do-we-start-a-new-notebook) to decide whether to create a new notebook or expand an existing one)

- **High-level Overview**:
    - **Problem Definition**: Describe the specific problem or challenge this task addresses.
    - **Comparison**: (Optional) Highlight how this task stands out or differs from other tasks in our codebase.
- **Detailed Walkthrough**:
    - **Input/Output Specifications**: Clearly define the expected input format and the resulting output.
    Mention any constraints or specific requirements.
    - **Debugging Insights**: Explain what information is available in the trace and how it can aid in troubleshooting.
    - **Use-case Examples**: What are concrete use-cases I can solve with this task?
    Run through examples.
    - **Evaluation Metrics**: (Optional) Suggest methods or metrics to evaluate the performance or accuracy of this task.

### How-tos

**Purpose**: How-tos are a short and concise way of understanding a very specific concept. They focus on a single thing which they go through in much detail in a step by step guide. 

- **Table of Content**: Which steps are covered in the how-to?
- **Detailed Walkthrough**: Guide the user step by step. Keep it short and concise.


### Docstrings

**Purpose**: Docstrings give a quickstart overview. They provide the necessary information for a user to be able to use this class/function in a correct manner. Not more, not less.

- **Summary**:
    - **One-Liner**: What does this class/function do?
    - **Brief Description**: What actually happens when I run this? What are need-to-know specifics?
- **Implementation Specifics**:
    - **Parameters & Their Significance**: List all parameters the class/function accepts.
    For each parameter, describe its role and why it's necessary.
    - **Requirements & Limitations**: What does this parameter require?
    Are there any limitations, such as text length?
    Is there anything else a user must know to use this?
    - **Usage Guidelines**: (Optional) Provide notes, insights or warnings about how to correctly use this class/function.
    Mention any nuances, potential pitfalls, or best practices.

By maintaining clear distinctions between the three documentation streams, we ensure that both users and developers have the necessary tools and information at their disposal for efficient task execution and code modification.

## Building a new task

To make sure that we approach building new tasks in a unified way, consider this example task:

``` python
# Standard import pattern:
# - Sort imports alphabetically.
# - Avoid wildcrad imports.
# - Normally consists of three blocks, separated by one newline each.
# 1) Built-in libraries
import math
from typing import Sequence

# 2) Third-party libraries
from intelligence_layer.connectors.limited_concurrency_client import LimitedConcurrencyClient
from pydantic import BaseModel
import requests # type: ignore
# Use 'type: ignore' for libraries that cause mypy issues (if there's no other fix).

# 3) Local application libraries
from intelligence_layer.examples.nested_task import NestedTask
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import Tracer


# Two newlines in between separate classes or between classes and imports.
class ExampleTaskInput(BaseModel):
    """Some documentation should go here. For information on this, see below."""
    # Each task can receive its required input in one of three ways:
    # 1) In the input of the task:
    # - The only parameters here should be the really "dynamic" ones, i.e. the ones that change from run to run.
    # -> As a rule of thumb: For QA, a query will change each run, a model will not.
    some_query: str
    some_number: int


class ExampleTaskOutput(BaseModel):
    """Some documentation should go here as well."""
    some_result: str


class ExampleTask(Task[ExampleTaskInput, ExampleTaskOutput]):
    """Even more fun documentation."""
    # 2) As constants:
    # - Makes it clear that these exact parameters are required for the `Task` or at least are central to it.
    # - Sends a clear signal that these parameters should not be touched.
    CONSTANT_PROMPT = "This prompt defines this task and its sole reason for existing."
    CONSTANT_MODEL = "unique_feature_llm" # only works with this model and no other

    # 3) In the `__init__`:
    # - Used for non-dynamic parameters, that stay the same for each task but may differ for task instances.
    # - Used for parameters that are initialized some time before and handed down/reused, such as the AA client.
    def __init__(
        init_model: str,
        init_client: AlephAlphaClientProtocol
    ) -> None:
        super().__init__()
        # In general: most attributes should be private, unless there is a specific reason for them being public.
        self._init_model = init_model # Used if multiple models can be used.
        self._init_client = init_client # Client should only be instantiated once, therefore the proper place is here.
        self._nested_task = NestedTask(init_client) # Try instantiating all tasks in the `__init__`, rather than in `run` or elsewhere.

    # For now, we assume that run will be the only explicitly public method for each `Task`.
    # `run` should be the first method after dunder methods.
    def run(self, input: ExampleTaskInput, tracer: Tracer) -> ExampleOutput:
        return self._some_calculation(input.some_number)

    # Example for a private method.
    # All such methods follow after `run`.
    def _some_calculation(some_number: int) -> float:
        return math.exp(some_number)
```

## When to use a tracer

Each task's input and output are automatically logged.
For most task, we assume that this suffices.

Exceptions would be complicated, task-specific implementations.
An example would the classify logprob calculation.

## When do we start a new notebook?

Documenting our LLM-based Tasks using Jupyter notebooks is crucial for clarity and ease of use.
However, we must strike a balance between consolidation and over-segmentation.
Here are the guidelines to determine when to start a new notebook:

- **Unified Purpose**: If a group of tasks shares a common objective or serves a similar function, they should be documented together in a single notebook.
This avoids redundancy and provides users with a centralized resource.
For instance, if there are multiple tasks all related to similar clasifications, they may be grouped.
- **Complexity & Length**: If detailing a task or a group of tasks would result in an exceedingly long or complex notebook, it's advisable to split them.
Each notebook should be digestible and focused, ensuring that users don't get overwhelmed.
- **Distinct Usage Scenarios**: If tasks have distinctly different use cases or are applied in separate stages of a project, they should have individual notebooks.
This ensures that users can quickly find and reference the specific task they need without sifting through unrelated content.
- **Interdependence**: Tasks that are interdependent or are typically used in tandem should be documented together.
This offers users a streamlined guide on how to use them in sequence or conjunction.
For example: `SingleChunkQA` & `TextHighlight`.
- **Feedback & Updates**: If a particular task receives frequent updates or modifications, it might be beneficial to keep it separate.
This ensures that changes to one task don't clutter or complicate the documentation of others.

In summary, while our goal is to keep our documentation organized and avoid excessive fragmentation, we must also ensure that each notebook is comprehensive and user-friendly.
When in doubt, consider the user's perspective: Would they benefit from a consolidated guide, or would they find it easier to navigate separate, focused notebooks?
