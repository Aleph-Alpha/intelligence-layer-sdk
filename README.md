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

### Table of contents

1. [Getting Started](#getting-started)
2. [Getting started with the Jupyter Notebooks](#getting-started-with-the-jupyter-notebooks)
3. [How to use this in your project](#how-to-make-your-own-use-case)
4. [Use-case index](#use-case-index)
5. [How to make your own use-case](#how-to-make-your-own-use-case)
6. [License](#license)

### How to make your own

Note that we do not expect the above use cases to solve all of your issues.
Instead, we encourage you to think of our pre-configured use cases as a foundation to fast-track your development process.
By leveraging these tasks, you gain insights into the framework's capabilities and best practices.

We encourage you to copy and paste these use cases directly into your own project.
From here, you can customize everything, including the prompt, model, and more intricate functional logic.
This not only saves you time but also ensures you're building on a tried and tested foundation.
Therefore, think of these use-cases as stepping stones, guiding you towards crafting tailored solutions that best fit your unique requirements.

## Getting started

Not sure where to start? Familiarize yourself with the Intelligence Layer using the below notebooks.

| Order | Topic              | Description                               | Notebook 📓                                                   |
| ----- | ------------------ | ----------------------------------------- | ------------------------------------------------------------- |
| 1     | Summarization      | Summarize a document                      | [summarize.ipynb](./src/examples/summarize.ipynb)             |
| 2     | Question Answering | Various approaches for QA                 | [qa.ipynb](./src/examples/qa.ipynb)                           |
| 3     | Quickstart task    | Build a custom task for your use case     | [quickstart_task.ipynb](./src/examples/quickstart_task.ipynb) |
| 4     | Classification     | Learn about two methods of classification | [classification.ipynb](./src/examples/classification.ipynb)   |
| 5     | Evaluation         | Evaluate LLM-based methodologies          | [evaluation.ipynb](./src/examples/evaluation.ipynb)           |
| 6     | Document Index     | Connect your proprietary knowledge base   | [document_index.ipynb](./src/examples/document_index.ipynb)   |

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

## Use-case index

To give you a starting point for using the Intelligence Layer, we provide some pre-configured `Task`s that are ready to use out-of-the-box, as well as an accompanying "Getting started" guide in the form of Jupyter Notebooks.

| Type      | Task                                                                                                                                                            | Description                                                                                                                                                                                                        |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Classify  | [EmbeddingBasedClassify](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.EmbeddingBasedClassify) | Classify a short text by computing its similarity with example texts for each class.                                                                                                                               |
| Classify  | [SingleLabelClassify](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SingleLabelClassify)       | Classify a short text by assessing each class' probability using zero-shot prompting.                                                                                                                              |
| QA        | [LongContextQa](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.LongContextQa)                   | Answer a question based on one document of any length.                                                                                                                                                             |
| QA        | [MultipleChunkQa](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.MultipleChunkQa)               | Answer a question based on a list of short texts.                                                                                                                                                                  |
| QA        | [RetrieverBasedQa](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.RetrieverBasedQa)             | Answer a question based on a document base using a [BaseRetriever](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation. |
| QA        | [SingleChunkQa](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.SingleChunkQa)                   | Answer a question based on a short text.                                                                                                                                                                           |
| Search    | [Search](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.Search)                                 | Search for texts in a document base using a [BaseRetriever](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.connectors.html#intelligence_layer.connectors.BaseRetriever) implementation.        |
| Summarize | [ShortBodySummarize](https://glowing-tribble-223446r.pages.github.io/intelligence_layer.use_cases.html#intelligence_layer.use_cases.ShortBodySummarize)         | Condense a short text into a brief summary.                                                                                                                                                                        |

## How to make your own use-case

Note that we do not expect the above use cases to solve all of your issues.
Instead, we encourage you to think of our pre-configured use cases as a foundation to fast-track your development process.
By leveraging these tasks, you gain insights into the framework's capabilities and best practices.

We encourage you to copy and paste these use cases directly into your own project.
From here, you can customize everything, including the prompt, model, and more intricate functional logic.
This not only saves you time but also ensures you're building on a tried and tested foundation.
Therefore, think of these use-cases as stepping stones, guiding you towards crafting tailored solutions that best fit your unique requirements.

## License

This project can only be used after signing the agreement with Aleph Alpha®. Please refer to the [LICENSE](LICENSE.md) file for more details.
