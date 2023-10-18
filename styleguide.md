# Intelligence Layer Style Guideline

## Documentation

Generally, adhere to this [guideline](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

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
