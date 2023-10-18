# Intelligence Layer☯️

Intelligence Layer☯️ is a suite of development tools designed by [Aleph Alpha](https://aleph-alpha.com/) enabling you to build end-to-end solutions utilizing the power of large language models (LLMs). Our goal here is to empower you to apply LLMs to solve various practical problems by providing a consistent **framework for designing LLM-based workflows**. The tools should enable you to go quickly through all the phases of AI product development, from prototyping and playing with prompts, through setting up experiments and evaluating their results, to solution deployment.

The key features of the intelligence layer are:

- **Best practices** - we give you access to *state-of-the-art* methods for addressing some of the most popular LLM use cases. You get access to off-the-shelf methods, enabling you to quickly build the first prototypes based on your primary data. We build on top of the best industry practices, optimizing the last bit of performance from LLMs.

- **Composability**: The Intelligence Layer enables you to easily go from the prototyping phase to a scallable deployed solution. We seamlessly integrate with various evaluation methods, handle concurrency, and compose smaller [Tasks](./src/intelligence_layer/task.py) (The task is at the very bottom of this file, maybe we should rather link into the documentation (which is not yet available, I know)) into more complicated workflows.

- **Auditability** The foundation assumption behind the Intelligence Layer is to give you access to the internal states of a [Task](./src/intelligence_layer/task.py) at every step of a workflow execution. This enables you to easier debug a [Task](./src/intelligence_layer/task.py) and gives you more control post deployment when you want to investigate how the model replies were produced.

## Tasks

Out of the box you get access to the following tasks:



| Task                | Description                                   | Notebook📓                                       |
|---------------------|-----------------------------------------------|------------------------------------------------|
| Summarization       | Use an LLM to summarize                       | [summarize.ipynb](./src/examples/summarize.ipynb)   |
| Question Answering  | Various approaches for QA                     | [qa.ipynb](./src/examples/qa.ipynb)        |
| Quickstart task         | We show you how to build a Task from scratch for your own custom use case | [quickstart_task.ipynb](./src/examples/quickstart_task.ipynb) |
| Classification      | Use an LLM to conduct zero-shot text classification. | [classify.ipynb](./src/examples/classify.ipynb) |


## Quickstart


**(Where does this come from? I guess one needs to clone right? And why would I cp afterwards? I only need to do this if I want to customize a task, right? But not necessarily for a quickstart?)**

Set up the poetry environment

```cmd
poetry install
```

Set up the Aleph Alpha token value

```cmd
export AA_TOKEN = <YOUR TOKEN HERE>
```

Run the `jupytyer lab`, and go to the [examples](http://localhost:8888/lab/workspaces/auto-C/tree/src/examples) dir.

```cmd
jupyter lab
```



## Development

Install [pre-commit](https://pre-commit.com/)
```cmd
pre-commit install
```

Run the CI scripts to ensure the typing is correct, linting follows the pattern, tests and jupyter notebooks run.

```cmd
scripts/precommit-and-mypy-and-pytest.sh
```
```cmd
scripts/notebook_runner.sh
```
