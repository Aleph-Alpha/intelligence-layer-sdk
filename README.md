# Aleph Alpha Intelligence Layer

The Aleph Alpha Intelligence Layer️ offers a comprehensive suite of development tools for crafting solutions that harness the capabilities of large language models (LLMs).
With a unified framework for LLM-based workflows, it facilitates seamless AI product development, from prototyping and prompt experimentation to result evaluation and deployment.

The key features of the Intelligence Layer are:

- **Best practices:** We provide you with **state-of-the-art** methods tailored for prevalent LLM use cases.
  Utilize our off-the-shelf techniques to swiftly prototype based on your primary data.
  Our approach integrates the best industry practices, allowing for optimal performance.
- **Composability:** The Intelligence Layer streamlines your journey from prototyping to scalable deployment.
  It offers seamless integration with diverse evaluation methods, manages concurrency, and orchestrates smaller tasks into complex workflows.
- **Auditability:** At the core of the Intelligence Layer is the belief that all AI processes must be auditable and traceable.
  To ensure this, we provide full comprehensibility, by seamlessly logging each step of every workflow.
  This enhances your debugging capabilities and offers greater control post-deployment when examining model responses.

# Table of contents
- [Aleph Alpha Intelligence Layer](#aleph-alpha-intelligence-layer)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
  - [Local installation (for development and tutorials)](#local-installation-for-development-and-tutorials)
    - [Getting started with the Jupyter Notebooks](#getting-started-with-the-jupyter-notebooks)
  - [How to use the Intelligence Layer in your project](#how-to-use-the-intelligence-layer-in-your-project)
  - [How to use the Intelligence Layer in Docker](#how-to-use-the-intelligence-layer-in-docker)
- [Getting started](#getting-started)
  - [Tutorials](#tutorials)
  - [How-Tos](#how-tos)
- [Use-case index](#use-case-index)
- [References](#references)
- [License](#license)
- [For Developers](#for-developers)
  - [Python: Naming Conventions](#python-naming-conventions)
# Installation
## Local installation (for development and tutorials)
Clone the Intelligence Layer repository from github.
```bash
git clone git@github.com:Aleph-Alpha/intelligence-layer.git
```
The Intelligence Layer uses `poetry` as a package manager. Follow the [official instructions](https://python-poetry.org/docs/#installation) to install it.
Afterwards, simply run `poetry install` to install all dependencies in a virtual environment.
```bash
poetry install
```
The environment can be activated via `poetry shell`. See the official poetry documentation for more information.


### Getting started with the Jupyter Notebooks

After running the local installation steps, there are two environment variables that have to be set before you can start running the examples.

---
**Using the Aleph-Alpha API** \
  \
You will need an [Aleph Alpha access token](https://docs.aleph-alpha.com/docs/account/#create-a-new-token) to run the examples.
Set your access token with

```bash
export AA_TOKEN=<YOUR TOKEN HERE>
```

---

**Using an on-prem setup** \
  \
The default host url in the project is set to `https://api.aleph-alpha.com`. This can be changed by setting the `CLIENT_URL` environment variable:

```bash
export CLIENT_URL=<YOUR_ENDPOINT_URL_HERE>
```

The program will warn you if no `CLIENT_URL` is explicitly set.

---
After correctly setting up the environment variables you can run the jupyter notebooks.
For this, run `jupyter lab` inside the virtual environment and go to the [examples](http://localhost:8888/lab/workspaces/auto-C/tree/src/examples) directory.

```bash
cd src/examples && poetry run jupyter lab
```

## How to use the Intelligence Layer in your project
To install this as a dependency in your project, you need a [Github access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic).
This token needs the following permissions

- `repo`
- `read:packages`

Set your access token:

```bash
export GITHUB_TOKEN=<YOUR_GITHUB_TOKEN>
```

[](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)

We recommend setting up a dedicated virtual environment. You can do so by using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [venv](https://docs.python.org/3/library/venv.html) or [poetry](https://python-poetry.org/).

You can add `intelligence-layer` to poetry dependencies. To do so, due to the limitations of poetry, you will need to modify the `git config` to make use of the `GITHUB_TOKEN`. To do so, prior to `poetry install`, run:

```bash
git config --global url."https://${GITHUB_TOKEN}@github.com/Aleph-Alpha/intelligence-layer".insteadOf "https://github.com/Aleph-Alpha/intelligence-layer"
```

after that add

```toml
[tool.poetry.dependencies]
python = ">=3.10,<3.12"
intelligence-layer = { git = "https://github.com/Aleph-Alpha/intelligence-layer.git", tag = "VERSION_TAG"}
```

to your `pyproject.toml` and run `poetry update`

Alternatively you can also add it to a `requirements.txt`.

```txt
git+https://$GITHUB_TOKEN@github.com/Aleph-Alpha/intelligence-layer.git@VERSION_TAG
```

Finally you can also install the package manually using pip.

```bash
pip install git+https://$GITHUB_TOKEN@github.com/Aleph-Alpha/intelligence-layer.git@VERSION_TAG
```

Now the Intelligence Layer should be available as a Python package and ready to use.

```py
from intelligence_layer.core import Task
```

In VSCode, to enable auto-import up to the second depth, where all symbols are exported, add the following entry to your `./.vscode/settings.json`:

``` json
"python.analysis.packageIndexDepths": [
    {
        "name": "intelligence_layer",
        "depth": 2
    }
]
```
## How to use the Intelligence Layer in Docker

To use the Intelligence Layer in Docker, a few settings are needed to not leak your Github token.

You will need your Github token set in your environment.

In order to modify the `git config` add the following to your docker container:

```dockerfile
RUN apt-get -y update
RUN apt-get -y install git curl gcc python3-dev
RUN pip install poetry

RUN --mount=type=secret,id=GITHUB_TOKEN \
    git config --global url."https://$(cat /run/secrets/GITHUB_TOKEN)@github.com/Aleph-Alpha/intelligence-layer".insteadOf "https://github.com/Aleph-Alpha/intelligence-layer" \
    && poetry install --no-dev --no-interaction --no-ansi \
    &&  rm -f ~/.gitconfig
```

Then to build your container, use the following command:

```bash
GITHUB_TOKEN=$GITHUB_TOKEN docker build --secret id=GITHUB_TOKEN <PATH_TO_DOCKERFILE>
```

If using a Docker compose file, add the following to your `docker-compose.yml`:

```yaml
services:
  service-using-intelligence-layer:
    build:
      context: .
      secrets:
        - GITHUB_TOKEN

secrets:
  GITHUB_TOKEN:
    # Needs to be set in your environment (.env) under the same name.
    environment: "GITHUB_TOKEN"
```

You can read more about this in the [official documentation](https://docs.docker.com/engine/swarm/secrets/).

# Getting started

Not sure where to start? Familiarize yourself with the Intelligence Layer using the below notebook as interactive tutorials.
If you prefer you can also read about the [concepts](Concepts.md) first.

## Tutorials
The tutorials aim to guide you through implementing several common use-cases with the Intelligence Layer. They introduce you to key concepts and enable you to create your own use-cases.

| Order | Topic              | Description                                          | Notebook 📓                                                      |
| ----- | ------------------ | ---------------------------------------------------- | --------------------------------------------------------------- |
| 1     | Summarization      | Summarize a document                                 | [summarization.ipynb](./src/examples/summarization.ipynb)       |
| 2     | Question Answering | Various approaches for QA                            | [qa.ipynb](./src/examples/qa.ipynb)                             |
| 3     | Classification     | Learn about two methods of classification            | [classification.ipynb](./src/examples/classification.ipynb)     |
| 4     | Evaluation         | Evaluate LLM-based methodologies                     | [evaluation.ipynb](./src/examples/evaluation.ipynb)             |
| 5     | Quickstart Task    | Build a custom `Task` for your use case              | [quickstart_task.ipynb](./src/examples/quickstart_task.ipynb)   |
| 6     | Document Index     | Connect your proprietary knowledge base              | [document_index.ipynb](./src/examples/document_index.ipynb)     |
| 7     | Human Evaluation   | Connect to Argilla for manual evaluation             | [human_evaluation.ipynb](./src/examples/human_evaluation.ipynb) |
| 8     | Performance tips   | Contains some small tips for performance             | [performance_tips.ipynb](./src/examples/performance_tips.ipynb) |
| 9     | Deployment         | Shows how to deploy a Task in a minimal FastAPI app. | [fastapi_example.py](./src/examples/fastapi_example.py)         |

## How-Tos
The how-tos are quick lookups about how to do things. Compared to the tutorials, they are shorter and do not explain the concepts they are using in-depth.

| Tutorial                                                                            | Description                                                             |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Tasks** | ---|
| [...define a task](./src/examples/how-tos/how_to_define_a_task.ipynb)               | How to come up with a new task and formulate it                         |
| [...implement a task](./src/examples/how-tos/how_to_implement_a_task.ipynb)         | Implement a formulated task and make it run with the Intelligence Layer |
| [...debug and log a task](./src/examples/how-tos/how_to_log_and_debug_a_task.ipynb) | Tools for logging and debugging in tasks                                |
| [...run the trace viewer](./src/examples/how-tos/how_to_run_the_trace_viewer.ipynb) | Downloading and running the trace viewer for debugging traces           |
| **Evaluation** | ---|
| [...implement a simple evaluation and aggregation logic](./src/examples/how-tos/how_to_implement_a_simple_evaluation_and_aggregation_logic.ipynb) | Basic use-case for evaluation and aggregation logic          |


# Use-case index

To give you a starting point for using the Intelligence Layer, we provide some pre-configured `Task`s that are ready to use out-of-the-box, as well as an accompanying "Getting started" guide in the form of Jupyter Notebooks.

| Type      | Task                                                                                                                                                                                                  | Description                                                                                                                                                                                                                                |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Classify  | [EmbeddingBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.EmbeddingBasedClassify)               | Classify a short text by computing its similarity with example texts for each class.                                                                                                                                                       |
| Classify  | [PromptBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.PromptBasedClassify)                     | Classify a short text by assessing each class' probability using zero-shot prompting.                                                                                                                                                      |
| Classify  | [KeywordExtract](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.KeywordExtract)                               | Generate matching labels for a short text.                                                                                                                                                                                                 |
| QA        | [LongContextQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.LongContextQa)                                 | Answer a question based on one document of any length.                                                                                                                                                                                     |
| QA        | [MultipleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.MultipleChunkQa)                             | Answer a question based on a list of short texts.                                                                                                                                                                                          |
| QA        | [RetrieverBasedQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.RetrieverBasedQa)                           | Answer a question based on a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation. |
| QA        | [SingleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SingleChunkQa)                                 | Answer a question based on a short text.                                                                                                                                                                                                   |
| Search    | [Search](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.Search)                                               | Search for texts in a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation.        |
| Summarize | [SteerableLongContextSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SteerableLongContextSummarize) | Condense a long text into a summary with a natural language instruction.                                                                                                                                                                   |
| Summarize | [SteerableSingleChunkSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SteerableSingleChunkSummarize) | Condense a short text into a summary with a natural language instruction.                                                                                                                                                                  |
| Summarize | [RecursiveSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.RecursiveSummarize)                       | Recursively condense a text into a summary.                                                                                                                                                                                                |

Note that we do not expect the above use cases to solve all of your issues.
Instead, we encourage you to think of our pre-configured use cases as a foundation to fast-track your development process.
By leveraging these tasks, you gain insights into the framework's capabilities and best practices.

We encourage you to copy and paste these use cases directly into your own project.
From here, you can customize everything, including the prompt, model, and more intricate functional logic.
For more information, check the [tutorials](#tutorials) and the [how-tos](#how-tos)



# References
The full code documentation can be found in our read-the-docs [here](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/)

# License

This project can only be used after signing the agreement with Aleph Alpha®. Please refer to the [LICENSE](LICENSE.md) file for more details.

# For Developers

## Python: Naming Conventions

We follow the [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/).
In addition, there are the following naming conventions:
* Class method names:
  * Use only substantives for a method name having no side effects and returning some objects
    * E.g., `evaluation_overview` which returns an evaluation overview object
  * Use a verb for a method name if it has side effects and return nothing
    * E.g., `store_evaluation_overview` which saves a given evaluation overview (and returns nothing)
