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

## Table of contents

1. [Getting Started](#getting-started)
2. [Getting started with the Jupyter Notebooks](#getting-started-with-the-jupyter-notebooks)
3. [How to use this in your project](#how-to-use-this-in-your-project)
4. [Use-case index](#use-case-index)
5. [How to run the Intelligence Layer Starter App](#how-to-run-the-intelligence-layer-starter-app)
6. [How to make your own use case](#how-to-make-your-own-use-case)
7. [Running the Trace Viewer](#running-the-trace-viewer)
8. [Using the Intelligence Layer in Docker](#using-the-intelligence-layer-in-docker)
9. [For Developers](#for-developers)
10. [References](#references)
11. [License](#license)

## Getting started

Not sure where to start? Familiarize yourself with the Intelligence Layer using the below notebook as interactive tuttorials.
If you prefer you can also read about the [concepts](Concepts.md) first.

| Order | Topic              | Description                               | Notebook 📓                                                     |
| ----- | ------------------ | ----------------------------------------- | --------------------------------------------------------------- |
| 1     | Summarization      | Summarize a document                      | [summarization.ipynb](./src/examples/summarize.ipynb)           |
| 2     | Question Answering | Various approaches for QA                 | [qa.ipynb](./src/examples/qa.ipynb)                             |
| 3     | Classification     | Learn about two methods of classification | [classification.ipynb](./src/examples/classification.ipynb)     |
| 4     | Evaluation         | Evaluate LLM-based methodologies          | [evaluation.ipynb](./src/examples/evaluation.ipynb)             |
| 5     | Quickstart Task    | Build a custom `Task` for your use case   | [quickstart_task.ipynb](./src/examples/quickstart_task.ipynb)   |
| 6     | Document Index     | Connect your proprietary knowledge base   | [document_index.ipynb](./src/examples/document_index.ipynb)     |
| 7     | Human Evaluation   | Connect to Argilla for manual evaluation  | [human_evaluation.ipynb](./src/examples/human_evaluation.ipynb) |
| 8     | Performance tips   | Contains some small tips for performance  | [performance_tips.ipynb](./src/examples/performance_tips.ipynb) |


## Getting started with the Jupyter Notebooks

Depending on your setup, there are two environment variables that have to be set before you can start running the examples.

---
### Using the Aleph-Alpha API:
You will need an [Aleph Alpha access token](https://docs.aleph-alpha.com/docs/account/#create-a-new-token) to run the examples.
First, set your access token:

```bash
export AA_TOKEN=<YOUR TOKEN HERE>
```

### Using an on-prem setup:
The default host url in the project is set to `https://api.aleph-alpha.com`. This can be changed by setting the `CLIENT_URL` environment variable:

```bash
export CLIENT_URL=<YOUR_ENDPOINT_URL_HERE>
```

The program will warn you if no `CLIENT_URL` is explicitly set.

---
After correctly setting up the environment variables, install all the dependencies:

```bash
poetry install
```

Run `jupyter lab`, and go to the [examples](http://localhost:8888/lab/workspaces/auto-C/tree/src/examples) directory.

```bash
cd src/examples && poetry run jupyter lab
```

## How to use this in your project

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

after that add:

```toml
[tool.poetry.dependencies]
python = ">=3.10,<3.12"
intelligence-layer = { git = "https://github.com/Aleph-Alpha/intelligence-layer.git", tag = "VERSION_TAG"}
```

to your `pyproject.toml` and run `poetry update`

Alternatively you can also add it to a `requirements.txt`

```txt
git+https://$GITHUB_TOKEN@github.com/Aleph-Alpha/intelligence-layer.git@VERSION_TAG
```

Finally you can also install the package manually using pip

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

## Use-case index

To give you a starting point for using the Intelligence Layer, we provide some pre-configured `Task`s that are ready to use out-of-the-box, as well as an accompanying "Getting started" guide in the form of Jupyter Notebooks.

| Type      | Task                                                                                                                                                                                                                  | Description                                                                                                                                                                                                                                |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Classify  | [EmbeddingBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.EmbeddingBasedClassify)                               | Classify a short text by computing its similarity with example texts for each class.                                                                                                                                                       |
| Classify  | [PromptBasedClassify](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.PromptBasedClassify)                                     | Classify a short text by assessing each class' probability using zero-shot prompting.                                                                                                                                                      |
| Classify  | [KeywordExtract](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.KeywordExtract)                                               | Generate matching labels for a short text.                                                                                                                                                                                                 |
| QA        | [LongContextQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.LongContextQa)                                                 | Answer a question based on one document of any length.                                                                                                                                                                                     |
| QA        | [MultipleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.MultipleChunkQa)                                             | Answer a question based on a list of short texts.                                                                                                                                                                                          |
| QA        | [RetrieverBasedQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.RetrieverBasedQa)                                           | Answer a question based on a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation. |
| QA        | [SingleChunkQa](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SingleChunkQa)                                                 | Answer a question based on a short text.                                                                                                                                                                                                   |
| Search    | [Search](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.Search)                                                               | Search for texts in a document base using a [BaseRetriever](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation.        |
| Summarize | [SteerableLongContextSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SteerableLongContextSummarize)                 | Condense a long text into a summary with a natural language instruction.                                                                                                                                                                   |
| Summarize | [SteerableSingleChunkSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SteerableSingleChunkSummarize)                 | Condense a short text into a summary with a natural language instruction.                                                                                                                                                                  |
| Summarize | [RecursiveSummarize](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.use_cases.html#intelligence_layer.use_cases.RecursiveSummarize)                                       | Recursively condense a text into a summary.                                                                                                                                                                                                |


## How to run the Intelligence Layer Starter App

The Starter App contains some of the core use-cases we offer out of the box.

To run it, run the following command:

```python
python3 src/intelligence_layer/use_cases/intelligence_starter_app.py
```

Now your app is running and you can start sending requests!

```bash
curl localhost:8000/openapi.json
```

To learn more about this, have a look at the [`IntelligenceLayerApp`](https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/intelligence_layer.core.html#intelligence_layer.core.IntelligenceApp).

## How to make your own use case

Note that we do not expect the above use cases to solve all of your issues.
Instead, we encourage you to think of our pre-configured use cases as a foundation to fast-track your development process.
By leveraging these tasks, you gain insights into the framework's capabilities and best practices.

We encourage you to copy and paste these use cases directly into your own project.
From here, you can customize everything, including the prompt, model, and more intricate functional logic.
This not only saves you time but also ensures you're building on a tried and tested foundation.
Therefore, think of these use-cases as stepping stones, guiding you towards crafting tailored solutions that best fit your unique requirements.

## Running the Trace Viewer

Make sure you have your `GITHUB_TOKEN` env variable set the same as above for installing the Python package, and that it has the `read:packages` permission.

Then login to the container registry with docker:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

Run the container locally:

```bash
docker run -p 3000:3000 ghcr.io/aleph-alpha/intelligence-layer-trace-viewer:latest
```

Finally, visit `http://localhost:3000`, where you can upload a trace to interact with the data.

## Using the Intelligence Layer in Docker

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

## For Developers

### Python: Naming Conventions

We follow the [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/).
In addition, there are the following naming conventions:
* Class method names:
  * Use only substantives for a method name having no side effects and returning some objects
    * E.g., `evaluation_overview` which returns an evaluation overview object
  * Use a verb for a method name if it has side effects and return nothing
    * E.g., `store_evaluation_overview` which saves a given evaluation overview (and returns nothing)

## References

- Full documentation: https://aleph-alpha-intelligence-layer.readthedocs-hosted.com/en/latest/

## License

This project can only be used after signing the agreement with Aleph Alpha®. Please refer to the [LICENSE](LICENSE.md) file for more details.
