# Intelligence Layer Style Guideline

## Documentation

Generally, adhere to this [guideline](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

## Documentation Guide: Jupyter Notebooks vs. Docstrings

When documenting our codebase, we focus on two primary channels: Jupyter notebooks and docstrings.
The objective is to provide both a high-level understanding and detailed implementation specifics.
Here's how we differentiate and allocate content between them:

### Jupyter Notebooks

**Purpose**: Jupyter notebooks are used to provide a comprehensive overview and walkthrough of the tasks. They are ideal for understanding the purpose, usage, and evaluation of a task.

- **High-level Overview**:
    - **Problem Definition**: Describe the specific problem or challenge this task addresses.
    - **Comparison**: (Optional) Highlight how this task stands out or differs from other tasks in our codebase.
- **Detailed Walkthrough**:
    - **Input/Output Specifications**: Clearly define the expected input format and the resulting output. Mention any constraints or specific requirements.
    - **Debugging Insights**: Explain what information is available in the debug log and how it can aid in troubleshooting.
    - **Use-case Examples**: What are concrete use-cases I can solve with this task? Run through examples.
    - **Evaluation Metrics**: (Optional) Suggest methods or metrics to evaluate the performance or accuracy of this task.

### Docstrings

**Purpose**: Docstrings give a quickstart overview. They provide the necessary information for user to be able to use this class/function in a correct manner. Not more, not less.

- **Summary**:
    - **One-Liner**: What does this class/function do?
    - **Brief Description**: What actually happens when I run this? What are need-to-know specifics?
- **Implementation Specifics**:
    - **Parameters & Their Significance**: List all parameters the class/function accepts. For each parameter, describe its role and why it's necessary.
    - **Requirements & Limitations**: What does this parameter require? Are there any limitations, such as text length? Is there anything else a user must know to use this?
    - **Usage Guidelines**: (Optional) Provide notes, insights or warnings about how to correctly use this class/function. Mention any nuances, potential pitfalls, or best practices.

---

By maintaining clear distinctions between the two documentation streams, we ensure that both users and developers have the necessary tools and information at their disposal for efficient task execution and code modification.


## Jupyter notebooks

Notebooks shall be used in a tutorial-like manner to educate users about certain tasks, functionalities & more.

## Docstrings

### Task documentation

Document any `Task` like so:
``` python
class MyTask:
    """Start with a one-line description of the task, like this.

    Follow up with a more detailed description, outlining the purpose & general functioning of the task.

    Note:
        What is important? Does your task require a certain type of model? Any usage recommendations?

    Args:
        example_arg: Any parameter provided in the '__init__' of this task.

    Attributes:
        EXAMPLE_CONSTANT: Any constant that may be defined within the class.
        example_non_private_attribute: Any attribute defined within the '__init__' that is not private.

    Example:
        >>> var = "Describe here how to use this task end to end"
        >>> print("End on two newlines.")
        End on two newlines.

    """
```

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

### Module documentation

We **do not document the module**, as we assume imports like:
``` python
from intelligence_layer.completion import Completion
completion_task = Completion()
```
rather than:
``` python
from intelligence_layerimport completion
completion_task = completion.Completion()
```
This ensures that the documentation is easily accessible by hovering over the imported task.
