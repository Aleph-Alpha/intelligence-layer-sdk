# Aleph Alpha Intelligence Layer ☯️

The  Aleph Alpha Intelligence Layer ☯️ offers a comprehensive suite of development tools for crafting solutions that harness the capabilities of large language models (LLMs).
With a unified framework for LLM-based workflows, it facilitates seamless AI product development, from prototyping and prompt experimentation to result evaluation and deployment.

The key features of the Intelligence Layer are:

- **Best practices** We provide you with **state-of-the-art** methods tailored for prevalent LLM use cases.
  Utilize our off-the-shelf techniques to swiftly prototype based on your primary data.
  Our approach integrates the best industry practices, allowing for optimal performance.
- **Composability**: The Intelligence Layer streamlines your journey from prototyping to scalable deployment.
  It offers seamless integration with diverse evaluation methods, manages concurrency, and orchestrates smaller tasks into complex workflows.
- **Auditability** At the core of the Intelligence Layer is the belief that all AI processes must be auditable and traceable.
  To ensure this, we provide full comprehensibility, by seamlessly logging each step of every workflow.
  This enhances your debugging capabilities and offers greater control post-deployment when examining model responses.

## Getting Started

Not sure where to start? Familiarize yourself with the Intelligence Layer using the below notebooks.

| Order | Task                           | Description                             | Notebook 📓                                                                     |
| ----- | ------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------- |
| 1     | Summarization                  | Summarize a document                    | [summarize.ipynb](./src/examples/summarize.ipynb)                               |
| 2     | Question Answering             | Various approaches for QA               | [qa.ipynb](./src/examples/qa.ipynb)                                             |
| 3     | Quickstart task                | Build a custom task for your use case   | [quickstart_task.ipynb](./src/examples/quickstart_task.ipynb)                   |
| 4     | Single label Classification    | Conduct zero-shot text classification   | [single_label_classify.ipynb](./src/examples/single_label_classify.ipynb)       |
| 5     | Embedding based Classification | Classify texts on the basis of examples | [embedding_based_classify.ipynb](./src/examples/embedding_based_classify.ipynb) |
| 6     | Document Index                 | Connect your proprietary knowledge base | [document_index.ipynb](./src/examples/document_index.ipynb)                     |

## Getting started with the Jupyter Notebooks

You will need an [Aleph Alpha](https://docs.aleph-alpha.com/docs/account/#create-a-new-token) access token to run the examples.
First, set your access token:

```bash
export AA_TOKEN=<YOUR TOKEN HERE>
```

Then, install all the dependencies:

```bash
poetry install
```

Run `jupytyer lab`, and go to the [examples](http://localhost:8888/lab/workspaces/auto-C/tree/src/examples) directory.

```bash
cd src/examples && poetry run jupyter lab
```

## How to use this in your project

To install this as a dependency in your project, you need a [Github access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

Set your access token:

```bash
GITHUB_TOKEN=<YOUR_GITHUB_TOKEN>
```
We recommend setting up a dedicated virtual environment. You can do so by using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [venv](https://docs.python.org/3/library/venv.html).


Let's install the package:

```bash
pip install git+https://$GITHUB_TOKEN@github.com/aleph-alpha-intelligence-layer/intelligence-layer.git
```

Now the Intelligence Layer should be available as a Python package and ready to use.

```py
from intelligence_layer.core.task import Task
```
