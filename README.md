# Aleph Alpha Intelligence Layer

The Aleph Alpha Intelligence Layer️ offers a comprehensive suite of development tools for crafting solutions that harness the capabilities of large language models (LLMs).
With a unified framework for LLM-based workflows, it facilitates seamless AI product development, from prototyping and prompt experimentation to result evaluation and deployment.

The key features of the Intelligence Layer are:

- **Composability:** Streamline your journey from prototyping to scalable deployment. The Intelligence Layer SDK offers seamless integration with diverse evaluation methods, manages concurrency, and orchestrates smaller tasks into complex workflows.
- **Evaluability:** Continuously evaluate your AI applications against your quantitaive quality requirements. With the Intelligence Layer SDK you can quickly iterate on different solution strategies, ensuring confidence in the performance of your final product. Take inspiration from the provided evaluations for summary and search when building a custom evaluation logic for your own use case.
- **Traceability:** At the core of the Intelligence Layer is the belief that all AI processes must be auditable and traceable. We provide full observability by seamlessly logging each step of every workflow. This enhances your debugging capabilities and offers greater control post-deployment when examining model responses.
- **Examples:** Get started by following our hands-on examples, demonstrating how to use the Intelligence Layer SDK and interact with its API.



# Table of contents
- [Aleph Alpha Intelligence Layer](#aleph-alpha-intelligence-layer)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
  - [Local installation (for development and tutorials)](#local-installation-for-development-and-tutorials)
    - [Getting started with the Jupyter Notebooks](#getting-started-with-the-jupyter-notebooks)
  - [How to use the Intelligence Layer in your project](#how-to-use-the-intelligence-layer-in-your-project)
  - [How to use the Intelligence Layer in Docker](#how-to-use-the-intelligence-layer-in-docker)
    - [Via the GitHub repository](#via-the-github-repository)
- [Getting started](#getting-started)
  - [Tutorials](#tutorials)
  - [How-Tos](#how-tos)
- [Models](#models)
- [Example index](#example-index)
- [References](#references)
- [License](#license)
- [For Developers](#for-developers)
  - [Python: Naming Conventions](#python-naming-conventions)
  - [Executing tests](#executing-tests)

# Installation

## Local installation (for development and tutorials)
Clone the Intelligence Layer repository from github.
```bash
git clone git@github.com:Aleph-Alpha/intelligence-layer-sdk.git
```
The Intelligence Layer uses `poetry` as a package manager. Follow the [official instructions](https://python-poetry.org/docs/#installation) to install it.
Afterwards, simply run `poetry install` to install all dependencies in a virtual environment.
```bash
poetry install
```
The environment can be activated via `poetry shell`. See the official poetry documentation for more information.


### Getting started with the Jupyter Notebooks

After running the local installation steps, you can set whether you are using the Aleph-Alpha API or an on-prem setup via the environment variables.

---
**Using the Aleph-Alpha API** \
  \
In the Intelligence Layer the Aleph-Alpha API (`https://api.aleph-alpha.com`) is set as default host URL. However, you will need an [Aleph Alpha access token](https://docs.aleph-alpha.com/docs/account/#create-a-new-token) to run the examples.
Set your access token with

```bash
export AA_TOKEN=<YOUR TOKEN HERE>
```

---

**Using an on-prem setup** \
  \
In case you want to use an on-prem endpoint you will have to change the host URL by setting the `CLIENT_URL` environment variable:

```bash
export CLIENT_URL=<YOUR_ENDPOINT_URL_HERE>
```

The program will warn you in case no `CLIENT_URL` is set explicitly set.

---
After correctly setting up the environment variables you can run the jupyter notebooks.
For this, run `jupyter lab` inside the virtual environment and go to the [examples](http://localhost:8888/lab/workspaces/auto-C/tree/src/documentation) directory.

```bash
cd src/documentation && poetry run jupyter lab
```

## How to use the Intelligence Layer in your project
To install the Aleph-Alpha Intelligence Layer from the JFrog artifactory in you project, you need an artifactory identity token. To generate this, log into artifactory in
your browser and open the user menu by clicking in the top-right corner. Then select 'Edit Profile' from the resulting dropdown menu. On the following page you can generate
an identity token by clicking the respective button after entering you password. Save the token at some secure place, e.g. your password manager.

With the token generated, you have to add this information to your poetry setup via the following four steps. First, add the artifactory as a source to your project via
```bash
poetry source add --priority=explicit artifactory https://alephalpha.jfrog.io/artifactory/api/pypi/python/simple
```
Second, to install the poetry environment, export your JFrog username and the generated token (NOT your actual password)
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

To use the Intelligence Layer in Docker, a few settings are needed to not leak your Github token.

You will need your Github token set in your environment.

In order to modify the `git config` add the following to your docker container:

```dockerfile
RUN apt-get -y update
RUN apt-get -y install git curl gcc python3-dev
RUN pip install poetry

RUN poetry install --no-dev --no-interaction --no-ansi \
    &&  rm -f ~/.gitconfig
```

# Getting started

Not sure where to start? Familiarize yourself with the Intelligence Layer SDK using the below notebook as interactive tutorials.
If you prefer you can also read about the [concepts](Concepts.md) first.

## Tutorials
The tutorials aim to guide you through implementing several common use-cases with the Intelligence Layer SDK. They introduce you to key concepts and enable you to create your own use-cases. In general the tutorials are build in a way that you can simply hop into the topic you are most interested in. However, for starters we recommend to read through the `Summarization` tutorial first. It explains the core concepts of the intelligence layer in more depth while for the other tutorials we assume that these concepts are known.

| Order | Topic                | Description                                          | Notebook 📓                                                                              |
|-------|----------------------|------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1     | Summarization        | Summarize a document                                 | [summarization.ipynb](./src/documentation/summarization.ipynb)                           |
| 2     | Question Answering   | Various approaches for QA                            | [qa.ipynb](./src/documentation/qa.ipynb)                                                 |
| 3     | Classification       | Learn about two methods of classification            | [classification.ipynb](./src/documentation/classification.ipynb)                         |
| 4     | Evaluation           | Evaluate LLM-based methodologies                     | [evaluation.ipynb](./src/documentation/evaluation.ipynb)                                 |
| 5     | Elo QA Evaluation    | Evaluate QA tasks in an Elo ranking                  | [elo_qa_eval.ipynb](./src/documentation/elo_qa_eval.ipynb)  |
| 6     | Quickstart Task      | Build a custom `Task` for your use case              | [quickstart_task.ipynb](./src/documentation/quickstart_task.ipynb)                       |
| 7     | Document Index       | Connect your proprietary knowledge base              | [document_index.ipynb](./src/documentation/document_index.ipynb)                         |
| 8     | Human Evaluation     | Connect to Argilla for manual evaluation             | [human_evaluation.ipynb](./src/documentation/human_evaluation.ipynb)                     |
| 9     | Performance tips     | Contains some small tips for performance             | [performance_tips.ipynb](./src/documentation/performance_tips.ipynb)                     |
| 10    | Deployment           | Shows how to deploy a Task in a minimal FastAPI app. | [fastapi_tutorial.ipynb](./src/documentation/fastapi_tutorial.ipynb)                     |
| 11    | Issue Classification | Deploy a Task in Kubernetes to classify Jira issues  | [Found in adjacent repository](https://github.com/Aleph-Alpha/IL-Classification-Journey) |

## How-Tos
The how-tos are quick lookups about how to do things. Compared to the tutorials, they are shorter and do not explain the concepts they are using in-depth.

| Tutorial                                                                                                                                          | Description                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Tasks**                                                                                                                                         |                                                                            |
| [...define a task](./src/documentation/how_tos/how_to_define_a_task.ipynb)                                                                             | How to come up with a new task and formulate it                            |
| [...implement a task](./src/documentation/how_tos/how_to_implement_a_task.ipynb)                                                                       | Implement a formulated task and make it run with the Intelligence Layer    |
| [...debug and log a task](./src/documentation/how_tos/how_to_log_and_debug_a_task.ipynb)                                                               | Tools for logging and debugging in tasks                                   |
| [...run the trace viewer](./src/documentation/how_tos/how_to_run_the_trace_viewer.ipynb)                                                               | Downloading and running the trace viewer for debugging traces              |
| **Analysis Pipeline**                                                                                                                             |                                                                            |
| [...implement a simple evaluation and aggregation logic](./src/documentation/how_tos/how_to_implement_a_simple_evaluation_and_aggregation_logic.ipynb) | Basic examples of evaluation and aggregation logic                         |
| [...create a dataset](./src/documentation/how_tos/how_to_create_a_dataset.ipynb)                                                                       | Create a dataset used for running a task                                   |
| [...run a task on a dataset](./src/documentation/how_tos/how_to_run_a_task_on_a_dataset.ipynb)                                                         | Run a task on a whole dataset instead of single examples                   |
| [...evaluate multiple runs](./src/documentation/how_tos/how_to_evaluate_runs.ipynb)                                                                    | Evaluate (multiple) runs in a single evaluation                            |
| [...aggregate multiple evaluations](./src/documentation/how_tos/how_to_aggregate_evaluations.ipynb)                                                    | Aggregate (multiple) evaluations in a single aggregation                   |
| [...retrieve data for analysis](./src/documentation/how_tos/how_to_retrieve_data_for_analysis.ipynb)                                                   | Retrieve experiment data in multiple different ways                        |
| [...implement a custom human evaluation](./src/documentation/how_tos/how_to_human_evaluation_via_argilla.ipynb)                                        | Necessary steps to create an evaluation with humans as a judge via Argilla |

# Models

Currently, we support a bunch of models accessible via the Aleph Alpha API. Depending on your local setup, you may even have additional models available.

| Model                                                                                                                                                                     | Description                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LuminousControlModel](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.LuminousControlModel) | Any control-type model based on the first Luminous generation, specifically `luminous-base-control`, `luminous-extended-control` and `luminous-supreme-control`. Multilingual support.          |
| [Llama2InstructModel](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.Llama2InstructModel)   | Llama-2 based models prompted for one-turn instruction answering. Includes `llama-2-7b-chat`, `llama-2-13b-chat` and `llama-2-70b-chat`. Best suited for English tasks.                         |
| [Llama3InstructModel](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.Llama3InstructModel)   | Llama-3 based models prompted for one-turn instruction answering. Includes `llama-3-8b-instruct` and `llama-3-70b-instruct`. Best suited for English tasks and recommended over llama-2 models. |

# Example index

To give you a starting point for using the Intelligence Layer, we provide some pre-configured `Task`s that are ready to use out-of-the-box, as well as an accompanying "Getting started" guide in the form of Jupyter Notebooks.

| Type      | Task                                                                                                                                                                                                            | Description                                                                                                                                                                                                                                |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Classify  | [EmbeddingBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.EmbeddingBasedClassify)                         | Classify a short text by computing its similarity with example texts for each class.                                                                                                                                                       |
| Classify  | [PromptBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.PromptBasedClassify)                               | Classify a short text by assessing each class' probability using zero-shot prompting.                                                                                                                                                      |
| Classify  | [PromptBasedClassifyWithDefinitions](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.PromptBasedClassifyWithDefinitions) | Classify a short text by assessing each class' probability using zero-shot prompting. Each class is defined by a natural language description.                                                                                             |
| Classify  | [KeywordExtract](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.KeywordExtract)                                         | Generate matching labels for a short text.                                                                                                                                                                                                 |
| QA        | [MultipleChunkRetrieverQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.MultipleChunkRetrieverQa)                     | Answer a question based on an entire knowledge base. Recommended for most RAG-QA use-cases.                                                                                                                                                |
| QA        | [LongContextQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.LongContextQa)                                           | Answer a question based on one document of any length.                                                                                                                                                                                     |
| QA        | [MultipleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.MultipleChunkQa)                                       | Answer a question based on a list of short texts.                                                                                                                                                                                          |
| QA        | [SingleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SingleChunkQa)                                           | Answer a question based on a short text.                                                                                                                                                                                                   |
| QA        | [RetrieverBasedQa (deprecated)](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.RetrieverBasedQa)                        | Answer a question based on a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation. |
| Search    | [Search](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.Search)                                                         | Search for texts in a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation.        |
| Search    | [ExpandChunks](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.ExpandChunks)                                             | Expand chunks retrieved with a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation.                     |
| Summarize | [SteerableLongContextSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SteerableLongContextSummarize)           | Condense a long text into a summary with a natural language instruction.                                                                                                                                                                   |
| Summarize | [SteerableSingleChunkSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SteerableSingleChunkSummarize)           | Condense a short text into a summary with a natural language instruction.                                                                                                                                                                  |
| Summarize | [RecursiveSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.RecursiveSummarize)                                 | Recursively condense a text into a summary.                                                                                                                                                                                                |

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



## Executing tests
**In VSCode**
1. Sidebar > Testing
2. Select pytest as framework for the tests
3. Select `intelligence_layer/tests` as source of the tests

You can then run the tests from the sidebar.

**In a terminal**
In order to run a local proxy w.r.t. to the CI pipeline (required to merge) you can run
> scripts/all.sh

This will run linters and all tests.
The scripts to run single steps can also be found in the `scripts` folder.
