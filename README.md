# Aleph Alpha Intelligence Layer

The Aleph Alpha Intelligence Layer️ offers a comprehensive suite of development tools for crafting solutions that harness the capabilities of large language models (LLMs).
With a unified framework for LLM-based workflows, it facilitates seamless AI product development, from prototyping and prompt experimentation to result evaluation and deployment.

The key features of the Intelligence Layer are:

- **Composability:** Streamline your journey from prototyping to scalable deployment. The Intelligence Layer SDK offers seamless integration with diverse evaluation methods, manages concurrency, and orchestrates smaller tasks into complex workflows.
- **Evaluability:** Continuously evaluate your AI applications against your quantitative quality requirements. With the Intelligence Layer SDK you can quickly iterate on different solution strategies, ensuring confidence in the performance of your final product. Take inspiration from the provided evaluations for summary and search when building a custom evaluation logic for your own use case.
- **Traceability:** At the core of the Intelligence Layer is the belief that all AI processes must be auditable and traceable. We provide full observability by seamlessly logging each step of every workflow. This enhances your debugging capabilities and offers greater control post-deployment when examining model responses.
- **Examples:** Get started by following our hands-on examples, demonstrating how to use the Intelligence Layer SDK and interact with its API.



# Table of contents
- [Aleph Alpha Intelligence Layer](#aleph-alpha-intelligence-layer)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
  - [Local installation (for development and tutorials)](#local-installation-for-development-and-tutorials)
  - [Add the Intelligence Layer to your project dependencies](#add-the-intelligence-layer-to-your-project-dependencies)
  - [How to use the Intelligence Layer in Docker](#how-to-use-the-intelligence-layer-in-docker)
    - [Via the GitHub repository](#via-the-github-repository)
- [Getting started](#getting-started)
    - [Setup LLM access](#setup-llm-access)
  - [Tutorial Notebooks](#tutorial-notebooks)
  - [How-Tos](#how-tos)
- [Models](#models)
- [Example index](#example-index)
- [References](#references)
- [License](#license)
- [For Developers](#for-developers)
  - [How to contribute](#how-to-contribute)
  - [Executing tests](#executing-tests)

# Installation

## Local installation (for development and tutorials)

Clone the Intelligence Layer repository from GitHub.

```bash
git clone git@github.com:Aleph-Alpha/intelligence-layer-sdk.git
```

The Intelligence Layer uses `poetry`, which serves as the package manager and manages the virtual environments.
We recommend installing poetry globally, while still isolating it in a virtual environment, using pipx, following the [official instructions](https://python-poetry.org/docs/#installation).
Afterward, simply run `poetry install` to create a new virtual environment and install all project dependencies.

```bash
poetry install
```
The environment can be activated via `poetry shell`. See the official poetry documentation for more information.

## Add the Intelligence Layer to your project dependencies

To install the Aleph-Alpha Intelligence Layer from the JFrog artifactory in you project, you have to add this information to your poetry setup via the following four steps. First, add the artifactory as a source to your project via
```bash
poetry source add --priority=explicit artifactory https://alephalpha.jfrog.io/artifactory/api/pypi/python/simple
```
Second, to install the poetry environment, export your JFrog credentials to the environment
```bash
export POETRY_HTTP_BASIC_ARTIFACTORY_USERNAME=your@username.here
export POETRY_HTTP_BASIC_ARTIFACTORY_PASSWORD=your-token-here
```
Third, add the Intelligence Layer to the project
```bash
poetry add --source artifactory intelligence-layer
```
Fourth, execute
```bash
poetry install
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

### Via the GitHub repository

To use the Intelligence Layer in Docker, a few settings are needed to not leak your GitHub token.

You will need your GitHub token set in your environment.

In order to modify the `git config` add the following to your docker container:

```dockerfile
RUN apt-get -y update
RUN apt-get -y install git curl gcc python3-dev
RUN pip install poetry

RUN poetry install --no-dev --no-interaction --no-ansi \
    &&  rm -f ~/.gitconfig
```

# Getting started

> 📘 Not sure where to start? Familiarize yourself with the Intelligence Layer using the **below notebooks as interactive tutorials**.
> If you prefer you can also **read about the [concepts](Concepts.md)** first.

The tutorials aim to guide you through implementing several common use-cases with the Intelligence Layer. They introduce you to key concepts and enable you to create your own use-cases. In general the tutorials are build in a way that you can simply hop into the topic you are most interested in. However, for starters we recommend to read through the `Summarization` tutorial first. It explains the core concepts of the intelligence layer in more depth while for the other tutorials we assume that these concepts are known.

### Setup LLM access

The tutorials require access to an LLM endpoint. You can choose between using the Aleph Alpha API (`https://api.aleph-alpha.com`) or an on-premise setup by configuring the appropriate environment variables. To configure the environment variables, create a `.env` file in the root directory of the project and copy the contents of the `.env.sample` file into it.

To use the **Aleph Alpha API**, that is set as the default host URL, set the `AA_TOKEN` variable to your [Aleph Alpha access token,](https://docs.aleph-alpha.com/docs/account/#create-a-new-token) and you are good to go.

To use an **on-premises setup**, set the `CLIENT_URL` variable to your host URL.

## Tutorial Notebooks

| Order | Topic                  | Description                                          | Notebook 📓                                                                              |
|-------|------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1     | Summarization          | Summarize a document                                 | [summarization.ipynb](./src/documentation/summarization.ipynb)                           |
| 2     | Question Answering     | Various approaches for QA                            | [qa.ipynb](./src/documentation/qa.ipynb)                                                 |
| 3     | Classification         | Learn about two methods of classification            | [classification.ipynb](./src/documentation/classification.ipynb)                         |
| 4     | Evaluation             | Evaluate LLM-based methodologies                     | [evaluation.ipynb](./src/documentation/evaluation.ipynb)                                 |
| 5     | Parameter Optimization | Compare Task configuration for optimization          | [parameter_optimization.ipynb](./src/documentation/parameter_optimization.ipynb)         |
| 5     | Elo QA Evaluation      | Evaluate QA tasks in an Elo ranking                  | [elo_qa_eval.ipynb](./src/documentation/elo_qa_eval.ipynb)                               |
| 6     | Quickstart Task        | Build a custom `Task` for your use case              | [quickstart_task.ipynb](./src/documentation/quickstart_task.ipynb)                       |
| 7     | Document Index         | Connect your proprietary knowledge base              | [document_index.ipynb](./src/documentation/document_index.ipynb)                         |
| 8     | Human Evaluation       | Connect to Argilla for manual evaluation             | [human_evaluation.ipynb](./src/documentation/human_evaluation.ipynb)                     |
| 9     | Performance tips       | Contains some small tips for performance             | [performance_tips.ipynb](./src/documentation/performance_tips.ipynb)                     |
| 10    | Deployment             | Shows how to deploy a Task in a minimal FastAPI app. | [fastapi_tutorial.ipynb](./src/documentation/fastapi_tutorial.ipynb)                     |
| 11    | Issue Classification   | Deploy a Task in Kubernetes to classify Jira issues  | [Found in adjacent repository](https://github.com/Aleph-Alpha/IL-Classification-Journey) |

## How-Tos

The how-tos are quick lookups about how to do things. Compared to the tutorials, they are shorter and do not explain the concepts they are using in-depth.

| Tutorial                                                                                                                                               | Description                                                                |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Tasks**                                                                                                                                              |                                                                            |
| [...define a task](./src/documentation/how_tos/how_to_define_a_task.ipynb)                                                                             | How to come up with a new task and formulate it                            |
| [...implement a task](./src/documentation/how_tos/how_to_implement_a_task.ipynb)                                                                       | Implement a formulated task and make it run with the Intelligence Layer    |
| [...debug and log a task](./src/documentation/how_tos/how_to_log_and_debug_a_task.ipynb)                                                               | Tools for logging and debugging in tasks                                   |
| [...use Studio with traces](./src/documentation/how_tos/how_to_use_studio_with_traces.ipynb)                                                    | Submitting Traces to Studio for debugging                                  |
| **Analysis Pipeline**                                                                                                                                  |                                                                            |
| [...implement a simple evaluation and aggregation logic](./src/documentation/how_tos/how_to_implement_a_simple_evaluation_and_aggregation_logic.ipynb) | Basic examples of evaluation and aggregation logic                         |
| [...create a dataset](./src/documentation/how_tos/how_to_create_a_dataset.ipynb)                                                                       | Create a dataset used for running a task                                   |
| [...run a task on a dataset](./src/documentation/how_tos/how_to_run_a_task_on_a_dataset.ipynb)                                                         | Run a task on a whole dataset instead of single examples                   |
| [...resume a run after a crash](./src/documentation/how_tos/how_to_resume_a_run_after_a_crash.ipynb) | Resume a run after a crash or exception occurred |
| [...evaluate multiple runs](./src/documentation/how_tos/how_to_evaluate_runs.ipynb)                                                                    | Evaluate (multiple) runs in a single evaluation                            |
| [...aggregate multiple evaluations](./src/documentation/how_tos/how_to_aggregate_evaluations.ipynb)                                                    | Aggregate (multiple) evaluations in a single aggregation                   |
| [...retrieve data for analysis](./src/documentation/how_tos/how_to_retrieve_data_for_analysis.ipynb)                                                   | Retrieve experiment data in multiple different ways                        |
| [...implement a custom human evaluation](./src/documentation/how_tos/how_to_human_evaluation_via_argilla.ipynb)                                        | Necessary steps to create an evaluation with humans as a judge via Argilla |
| [...implement elo evaluations](./src/documentation/how_tos/how_to_implement_elo_evaluations.ipynb)                                                     | Evaluate runs and create ELO ranking for them                              |
| [...implement incremental evaluation](./src/documentation/how_tos/how_to_implement_incremental_evaluation.ipynb)                                       | Implement and run an incremental evaluation                                |

# Models

Currently, we support a bunch of models accessible via the Aleph Alpha API. Depending on your local setup, you may even have additional models available.

| Model                                                                                                                                                                     | Description                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [LuminousControlModel](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.LuminousControlModel) | Any control-type model based on the first Luminous generation, specifically `luminous-base-control`, `luminous-extended-control` and `luminous-supreme-control`.                                                                                   |
| [Pharia1ChatModel](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.Pharia1ChatModel)         | Pharia-1 based models prompted for multi-turn interactions. Includes `Pharia-1-LLM-7B-control` and `Pharia-1-LLM-7B-control-aligned`.                                                                                                              |
| [Llama3InstructModel](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.Llama3InstructModel)   | Llama-3 based models prompted for one-turn instruction answering. Includes `llama-3-8b-instruct`, `llama-3-70b-instruct`, `llama-3.1-8b-instruct` and `llama-3.1-70b-instruct`.                                                                    |
| [Llama3ChatModel](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.Llama3ChatModel)           | Llama-3 based models prompted for multi-turn interactions. Includes `llama-3-8b-instruct`, `llama-3-70b-instruct`, `llama-3.1-8b-instruct` and `llama-3.1-70b-instruct`.                                                                           |

# Example index

To give you a starting point for using the Intelligence Layer, we provide some pre-configured `Task`s that are ready to use out-of-the-box, as well as an accompanying "Getting started" guide in the form of Jupyter Notebooks.

| Type      | Task                                                                                                                                                                                                            | Description                                                                                                                                                                                                                                |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Classify  | [EmbeddingBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.EmbeddingBasedClassify)                         | Classify a short text by computing its similarity with example texts for each class.                                                                                                                                                       |
| Classify  | [PromptBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.PromptBasedClassify)                               | Classify a short text by assessing each class' probability using zero-shot prompting.                                                                                                                                                      |
| Classify  | [PromptBasedClassifyWithDefinitions](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.PromptBasedClassifyWithDefinitions) | Classify a short text by assessing each class' probability using zero-shot prompting. Each class is defined by a natural language description.                                                                                             |
| Classify  | [KeywordExtract](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.KeywordExtract)                                         | Generate matching labels for a short text.                                                                                                                                                                                                 |
| QA        | [MultipleChunkRetrieverQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.MultipleChunkRetrieverQa)                     | Answer a question based on an entire knowledge base. Recommended for most RAG-QA use-cases.                                                                                                                                                |
| QA        | [LongContextQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.LongContextQa)                                           | Answer a question based on one document of any length.                                                                                                                                                                                     |
| QA        | [MultipleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.MultipleChunkQa)                                       | Answer a question based on a list of short texts.                                                                                                                                                                                          |
| QA        | [SingleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.SingleChunkQa)                                           | Answer a question based on a short text.                                                                                                                                                                                                   |
| QA        | [RetrieverBasedQa (deprecated)](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.RetrieverBasedQa)                        | Answer a question based on a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation. |
| Search    | [Search](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.Search)                                                         | Search for texts in a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation.        |
| Search    | [ExpandChunks](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.ExpandChunks)                                             | Expand chunks retrieved with a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation.                     |
| Summarize | [SteerableLongContextSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.SteerableLongContextSummarize)           | Condense a long text into a summary with a natural language instruction.                                                                                                                                                                   |
| Summarize | [SteerableSingleChunkSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.SteerableSingleChunkSummarize)           | Condense a short text into a summary with a natural language instruction.                                                                                                                                                                  |
| Summarize | [RecursiveSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.examples.html#intelligence_layer.examples.RecursiveSummarize)                                 | Recursively condense a text into a summary.                                                                                                                                                                                                |

Note that we do not expect the above use cases to solve all of your issues.
Instead, we encourage you to think of our pre-configured use cases as a foundation to fast-track your development process.
By leveraging these tasks, you gain insights into the framework's capabilities and best practices.

We encourage you to copy and paste these use cases directly into your own project.
From here, you can customize everything, including the prompt, model, and more intricate functional logic.
For more information, check the [tutorials](#tutorial-notebooks) and the [how-tos](#how-tos)

# References

The full code documentation can be found in our read-the-docs [here](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/)

# License

This project can only be used after signing the agreement with Aleph Alpha®. Please refer to the [LICENSE](LICENSE.md) file for more details.

# For Developers

For further information check out our different guides and documentations:
- [Concepts.md](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/Concepts.md) for an overview of what Intelligence Layer is and how it works.
- [style_guide.md](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/style_guide.md) on how we write and document code.
- [RELEASE.md](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/RELEASE.md) for the release process of IL.
- [CHANGELOG.md](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/CHANGELOG.md) for the latest changes.

## How to contribute
:warning: **Warning:** This repository is open-source. Any contributions you make will be publicly accessible.


1. Share the details of your problem with us.
2. Write your code according to our [style guide](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/style_guide.md).
3. Add doc strings to your code as described [here](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/style_guide.md#docstrings).
4. Write tests for new features ([Executing Tests](#executing-tests)).
5. Add an how_to and/or notebook as a documentation (check out [this](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/style_guide.md#documentation) for guidance).
6. Update the [Changelog](https://github.com/Aleph-Alpha/intelligence-layer-sdk/blob/main/CHANGELOG.md) with your changes.
7. Request a review for the MR, so that it can be merged.



## Executing tests
If you want to execute all tests, you first need to spin up your docker container and execute the commands with your own `GITLAB_TOKEN`.

```bash
  export GITLAB_TOKEN=...
  echo $GITLAB_TOKEN | docker login registry.gitlab.aleph-alpha.de -u your_email@for_gitlab --password-stdin
  docker compose pull to update containers
```

 Afterwards simply run `docker compose up --build`. You can then either run the tests in your IDE or via the terminal.

**In VSCode**
1. Sidebar > Testing
2. Select pytest as framework for the tests
3. Select `intelligence_layer/tests` as source of the tests

You can then run the tests from the sidebar.

**In a terminal**
In order to run a local proxy of the CI pipeline (required to merge) you can run
> scripts/all.sh

This will run linters and all tests.
The scripts to run single steps can also be found in the `scripts` folder.
