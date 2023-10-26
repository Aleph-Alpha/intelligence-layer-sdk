# Aleph Alpha Intelligence Layer ☯️

The Aleph Alpha Intelligence Layer ☯️ offers a comprehensive suite of development tools for crafting solutions that harness the capabilities of large language models (LLMs).
With a unified framework for LLM-based workflows, it facilitates seamless AI product development, from prototyping and prompt experimentation to result evaluation and deployment.

The key features of the Intelligence Layer are:

- **Best practices** We provide you with **_state-of-the-art_** methods tailored for prevalent LLM use cases.
  Utilize our off-the-shelf techniques to swiftly prototype based on your primary data.
  Our approach integrates the best industry practices, allowing for optimal performance.
- **Composability**: The Intelligence Layer streamlines your journey from prototyping to scalable deployment.
  It offers seamless integration with diverse evaluation methods, manages concurrency, and orchestrates smaller tasks into complex workflows.
- **Auditability** At the core of the Intelligence Layer is the belief that all AI processes must be auditable and traceable.
  To ensure this, we provide full coprehensibility, by seamlessly logging each step of every workflow.
  This enhances your debugging capabilities and offers greater control post-deployment when examining model responses.

## Getting Started

Not sure where to start? Familiarize yourself with the Intelligence Layer using the below notebooks.

| Order | Task               | Description                             | Notebook 📓                                                   |
| ----- | ------------------ | --------------------------------------- | ------------------------------------------------------------- |
| 1     | Summarization      | Summarize a document                    | [summarize.ipynb](./src/examples/summarize.ipynb)             |
| 2     | Question Answering | Various approaches for QA               | [qa.ipynb](./src/examples/qa.ipynb)                           |
| 3     | Quickstart task    | Build a custom task for your use case   | [quickstart_task.ipynb](./src/examples/quickstart_task.ipynb) |
| 4     | Classification     | Conduct zero-shot text classification   | [classify.ipynb](./src/examples/classify.ipynb)               |
| 5     | Document Index     | Connect your proprietary knowledge base | [document_index.ipynb](./src/examples/document_index.ipynb)   |

## Quickstart Guide

First, you need an Aleph Alpha access token. [Learn more](https://docs.aleph-alpha.com/docs/account/#create-a-new-token).

Install the intelligence layer package.

```cmd
# pip
pip install .
# poetry
poetry install
```

Set the Aleph Alpha token value

```cmd
export AA_TOKEN=<YOUR TOKEN HERE>
```

Run `jupytyer lab`, and go to the [examples](http://localhost:8888/lab/workspaces/auto-C/tree/src/examples) dir.

```cmd
(cd src/examples) && jupyter lab
```
