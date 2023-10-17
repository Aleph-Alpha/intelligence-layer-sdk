# Intelligence Layer

Intelligence Layer is a suite of development tools designed by [Aleph Alpha](https://aleph-alpha.com/) enabling you to build end-to-end solutions utilizing the power of large language models (LLMs). Our goal here is to empower you to apply LLMs to solve various practical problems by providing a consistent framework for prompt engineering. The tool should enable you to go quickly through all the phases of AI product development, from prototyping and playing with prompts, through setting up experiments and evaluating their results, to solution deployment.

The key features of the intelligence layer are:

- **Best practices** - we give you access to *state-of-the-art* methods for addressing some of the most popular LLM use cases. You get access to off-the-shelf methods, enabling you to quickly build the first prototypes based on your primary data. We build on top of the best industry practices, optimizing the last bit of performance from LLMs.

- **Composability**: The Intelligence Layer enables you to easily go from the prototyping phase to a scallable deployed solution. We seamlessly integrate with various evaluation methods, handle concurrency, and compose smaller [Tasks](./src/intelligence_layer/task.py) into more complicated workflows.

- **Auditability** The foundation assumption behind the Intelligence Layer is to give you access to the internal states of a [Task](./src/intelligence_layer/task.py) at every step of a workflow execution. This enables you to easier debug a [Task](./src/intelligence_layer/task.py) and gives you more control post deployment when you want to investigate how the model replies were produced.

## Tasks

Out of the box you get access to the following tasks:



| Task                | Description                                   | Notebook                                       |
|---------------------|-----------------------------------------------|------------------------------------------------|
| Classification      | We show how to utilize | [Classification Notebook](https://example.com/classification) |
| Question Answering  | Task description for QA goes here.            | [QA Notebook](https://example.com/question-answering)        |
| Summarization       | Task description for Summarization goes here. | [Summarization Notebook](https://example.com/summarization)   |



## Quickstart


Copy the files from the `intelligence-layer`

```cmd
cp -r intelligence-layer my-intelligence-layer
```

```cmd
cd my-intelligence-layer
```

Set up the poetry environment

```cmd
poetry install
```

Set up the Aleph Alpha token value

```cmd
export AA_TOKEN = <YOUR TOKEN HERE>
```

Run the `jupytyer notebook`, and go to the `Examples` dir.

```cmd
poetry run jupyter notebook
```



## Development

Install [pre-commit](https://pre-commit.com/)
```cmd
pre-commit install
```

Run the CI scripts to ensure the typing is correct, linting follows the pattern, tests and jupyter notebooks run.

```cmd
chmod +x scripts/precommit-and-mypy-and-pytest.sh
```
```cmd
chmod +x scripts/notebook_runner.sh
```
```cmd
scripts/precommit-and-mypy-and-pytest.sh
```
```cmd
scripts/notebook_runner.sh
```
