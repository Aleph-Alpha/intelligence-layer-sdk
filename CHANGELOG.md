# Changelog

## Unreleased

- The elo-calculation logic has been heavily simplified
- `Payoff` from the elo package has been renamed to `Match`
- `PayoffMatrix` from the elo package has been renamed to `MatchOutcome` and is now pydantic (de)-serializable
- `SingleChunkQa` now uses a logit_bias to promote not answering for German
- `__init__`-parameters of all tasks are streamlined:
    - Sub-tasks are typically exposed as `__init__`-parameters with sensible defaults.
    - Defaults for non-trivial parameters like models or tasks are defined in `__init__` while the default parameter is `None`.
    - Instead of exposing parameters that are passed on to sub-tasks the sub-task themselves are exposed.
- `IntelligenceApp` and `IntelligenceStarterApp` have been removed.


## 0.6.0

### Breaking Changes

- breaking change: The evaluation module is moved from core to evaluation .
- breaking change: RetrieverBasedQa task answers now contain document ids in each subanswer
- breaking change: LongcontextSummarize no longer supports the max_loops parameter
- breaking change: Rich Mode Representation
    - The LLM-based tasks no longer accept client, but rather an  AlephAlphaModel, which holds the client. The available model classes are  AlephAlphaModel and LuminousControlModel
    - The AlephAlphaModel is responsible for its prompt format, tokenizers, complete task and explain task. These responsibilities were moved into the model classes.
    - The default client url is now configurable via the environment variable CLIENT_URL
- breaking change: PromptWithMetadata is removed in favor of RichPrompt . The semantics remain largely unchanged
- breaking change: The compression-dependent long context summarize classes as well as the few-shot summarize class were removed. Use the better-performing steerable summary classes.
- breaking change: Runner, Evaluator & Aggregation
    - The EvaluationRepository has been split up. There is now a total of four repositories: dataset , run, evaluation and aggregation. These repositories save information from their respective steps
    - The evaluation and evaluation aggregation have been split and are now provided by the classes Evaluator and Aggregator, respectively. These two classes have no abstract methods. The evaluation and aggregation logic is provided by implementing the abstract methods of the classes EvaluationLogic and AggregationLogic which are passed on to an instance of the Evaluator and Aggregator class, respectively. For an example, see the Jupyter notebook xxx.

### New Features

- Documentation
    - feature: Added an intro to the Intelligence Layer concepts in Concepts.md
    - feature: Added documentation on how to execute tasks in parallel. See the performance_tips notebook for more information.
- QA
    - feature: RetrieverBasedQa task no longer sources its final from all sources, but only the most relevant. This performed better in evaluation.
    - feature: The notebooks for RetrieverBasedQa have been updated to use SingleChunkQa.
    - feature: SingleChunkQa now supports a custom no-answer phrase
    - feature: MultiChunkQA and LongContextQa allow for more configuration of the used qa-task.
    - feature: Make the distance metric configurable in QdrantInMemoryRetriever.
    - features: Added list_namespaces to DocumentIndexClient to list all available namespaces in DocumentIndex.
- Evaluation
    - feature: The argilla now supports splitting a dataset for multiple people via the split_dataset function
    - feature: Utilities for ELO score/ranking calculation
        - The build_tournaments utility function has been added to facilitate the computation of ELO scores when evaluating two models. See InstructComparisonArgillaEvaluator for an example how it can be used to compute the ELO scores.
    - feature: The Evaluator can run multiple evaluation tasks in parallel.
- Intelligence app
    - feature: IntelligenceApp returns 204 if the output is None
    - feature: Allow registering tasks with a task dependency in IntelligenceApp.
- Others
    - feature: Runner accepts in run_dataset a new parameter num_examples specifying how many of the first n examples should be run.
    - feature: Support None as return type in Task
    - feature: Added a new task: ChunkOverlapTask splits a longer text into overlapping chunks.

## 0.5.1

Failed deploy

## 0.5.0

### Breaking Changes

- Document Index search results now properly return `DocumentChunk`s instead of `Document` objects to make it clear it is only a portion of the document.
- `Instruct` and `FewShot` tasks now take the model name in the constructor instead of the input.
- `Dataset`s have now been moved to `DatasetRepository`s, which are responsible for loading and storing datasets. This allows for more flexibility in how datasets are loaded and stored.

### New Features

- Introduced an `OpenTelemetryTracer` to allow for sending trace spans to an OpenTelemetry collector.
- Notebook walking through how to use Argilla for human evaluation
- `SteerableLongContextSummarize` task that allows for steering the summarization process by providing a natural language instruction.
- Document index `SearchResult`s now also return the document ID for each chunk, to make it easier to retrieve the full document.
- Retrievers now supply a way to retrieve the full document by ID.
- Introduced the concept of `Accumulator`s to evaluation for incrementally calculating metrics.
- Added `EloCalculator` metrics for calculating Elo scores in evaluation methods.
- Introduced new `HuggingFaceDatasetRepository` for loading datasets from the HuggingFace datasets library.
- Made it easier to evaluate two tasks and or models against each other.

### Fixes

- Argilla client properly handles pagination when retrieving records
- Ensured file-based repositories are writing and reading in UTF-8


## 0.4.1

Fix missing version bump in the packages

## 0.4.0

### Breaking Changes

- `Evaluator` methods changed to support asynchronous processing for human eval. To run everything at once, change `evaluator.evaluate()` calls to `evaluator.run_and_evaluate`
    - An evaluation also now returns a `EvaluationOverview`, with much more information about the output of the evaluation.
- `EmbeddingBasedClassify`: init arguments swapped places, from `labels_with_examples, client` to `client, label_with_examples`
- `PromptOutput` for `Instruct` tasks now inherits from `CompleteOutput` to make it easier to use more information about the raw completion response.

### New Features

- New `IntelligenceApp` builder to quickly spin up a FastAPI server with your `Task`s
- Integration with [Argilla](https://docs.argilla.io/en/latest/index.html) for human evaluation
- `CompleteOutput` and `PromptOutput` now support getting the `generated_tokens` in the completion for downstream calculations.
- Summarization use cases now allow for overriding the default model
- New `RecursiveSummarizer` allows for recursively calling one of the `LongContextSummarize` tasks until certain thresholds are reached

### Fixes

- `LimitedConcurrencyClient`'s `from_token` method now supports a custom API host

## 0.3.0:

### Breaking Changes

- `Dataset` is now a protocol. `SequenceDataset` replaces the old `Dataset`.
- The `ident` attribute on `Example` is now `id`.
- `calculate_bleu` function is removed and instead called from a `BleuGrader`
- `calculate_rouge` function is removed and instead called from a `RougeGrader`
- `ClassifyEvaluator` is now called `SingleLabelClassifyEvaluator`
- `Evaluator`s now take and return `Iterator`s instead of `Sequence`s to allow for streaming datasets

### New Features

- `Evaluators` now have better handling of dataset processing.
  - Errors are handled for individual examples, so that you don't lose the entire run because of one failed task.
  - The dataset run now produces an `EvaluationRunOverview` generated by an `EvaluationRepository`, that better captures the aggregated runs and traces.
  - There is a `FileEvaluationRepository` and an `InMemoryEvaluationRepository` available for storing your evaluation results
- Support passing `Metadata` field through `DocumentIndexClient` (already supported in the Document Index, new in client only)
- New `MultiLabelClassifyEvaluator` to evaluate classification use cases that support multi-label classification
- `Evaluators` can now be called via the CLI

### Fixes

- Fix issue in `EchoTask` regarding concurrent execution causing overrides in the `PromptTemplate`

## 0.2.0

### Breaking Changes

- `SingleLabelClassify` renamed to `PromptBasedClassify` with new `SingleLabelClassifyOutput`
- `EmbeddingBasedClassify` now outputs `MultiLabelClassifyOutput` to distinguish between the different types of scores produced

### New Features

- New `LimitedConcurrencyClient` to better control how many simultaneous API requests are made concurrently, regardless of where they are called within the Task hierarchy
- Basic new `SingleChunkSummarizeEvaluator` and `LongContextSummarizeEvaluator` that can calculate Rouge and Bleu scores when compared with a "golden summary"

### Fixes

- Fix issue with Pydantic 2.5 due to ambiguous ordering of types in `PydanticSerializable` type
- Fixed possible deadlock with nested calls to `Task.run_concurrently`

## 0.1.0

Initial release
