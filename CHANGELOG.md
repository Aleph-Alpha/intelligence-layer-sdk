# Changelog

## Unreleased

### Breaking Changes
- feature: `HuggingFaceDatasetRepository` now has a parameter `caching`, which caches a examples of a dataset once loaded. This is `True` by default. This drastically reduces network traffic. For non-breaking changes, set it to `False`.
- breaking_change: `MultipleChunkRetrieverQa` does not take `insert_chunk_size`-parameter but instead takes `ExpandChunks`-task
- breaking_change: the `issue_cassification_user_journey` notebook is moved to its own repository

### New Features
- feature: `Llama2InstructModel` to support llama-2 models in Aleph Alpha API
- feature: `Llama3InstructModel` to support llama-3 models in Aleph Alpha API
- feature: `ExpandChunks`-task caches chunked documents by ID
- feature: `DocumentIndexClient` now supports
            - `create_index`
            - `index_configuration`
            - `assign_index_to_collection`
            - `delete_index_from_collection`
            - `list_assigned_index_names`
- feature: `DocumentIndexRetriever` now supports `index_name`
- feature: `Runner.run_dataset` now has a configurable number of workers via `max_workers` and defaults to the previous value, which is 10.
- feature: In case a `BusyError` is raised during a `complete` the `LimitedConcurrencyClient` will retry until `max_retry_time` is reached.
- feature: `FileTracer` now accepts as `log_file_path` both, a `str` and a `Path`

### Fixes
- refactor: rename `index` parameter in `DocumentIndex.search()` to `index_name`
- fix: `HuggingFaceRepository` no longer is a dataset repository. This also means that `HuggingFaceAggregationRepository` no longer is a dataset repository.


### Deprecations
- `RetrieverBasedQa` is now deprecated and will be removed in future versions. We recommend using `MultipleChunkRetrieverQa` instead.

## 0.10.0

### Breaking Changes
- breaking change: `ExpandChunksOutput` now returns `ChunkWithStartEndIndices` instead of `TextChunk`
- breaking change: `MultipleChunkRetrieverQa`'s `AnswerSource` now contains `EnrichedChunk` instead of just the `TextChunk`
- breaking change: `DocumentIndexClient` method `asymmetric_search()` has been removed
- breaking change: `DocumentIndexRetriever` now additionally needs `index_name`

### New Features

### Fixes
- fix: `ChunkWithIndices` now additionally returns end_index
- fix: `DocumentPath` and `CollectionPath` are now immutable

## 0.9.1

### Breaking Changes
- breaking change: `MultipleChunkRetrieverQaOutput` now return `sources` and `search_results`

### New Features
- feature: `ExpandChunks` task takes a retriever and some search results to expand the chunks to the desired length

### Fixes
- fix: `ExpectedSearchOutput` has only relevant fields and supports generic document-`ID` rather than just str
- fix: `SearchEvaluationLogic` explicitly compares documents by ids
- fix: In `RecusrsiveSummarize.do_run`, `num_generated_tokens` not uninitialized anymore. [See Issue 743.](https://github.com/Aleph-Alpha/intelligence-layer/issues/743).
- fix: Reverted pydantic to 2.6.* because of FastAPI incompatibility.

## 0.9.0

### Breaking Changes
 - breaking change: Renamed the field `chunk` of `AnswerSource` to `search_result` for multi chunk retriever qa.
 - breaking change: The implementation of the HuggingFace repository creation and deletion got moved to `HuggingFaceRepository`

### New Features
 - feature: HuggingFaceDataset- & AggregationRepositories now have an explicit `create_repository` function.
 - feature: Add `MultipleChunkRetrieverBasedQa`, a task that performs better on faster on retriever-QA, especially with longer context models

## 0.8.2

### New Features
 - feature: Add `SearchEvaluationLogic` and `SearchAggregationLogic` to evaluate `Search`-use-cases
 - feature: Trace viewer and IL python package are now deployed to artifactory

### Fixes
 - Documentation
   - fix: Add missing link to `issue_classification_user_journey` notebook to the tutorials section of README.
   - fix: Confusion matrix in `issue_classification_user_journey` now have rounded numbers.

## 0.8.1

### Fixes
- fix: Linting for release version

## 0.8.0

### New Features
- feature: Expose start and end index in DocumentChunk
- feature: Add sorted_scores property to `SingleLabelClassifyOutput`.
- feature: Error information is printed to the console on failed runs and evaluations.
- feature: The stack trace of a failed run/evaluation is included in the `FailedExampleRun`/`FailedExampleEvaluation` object
- feature: The `Runner.run_dataset(..)` and `Evaluator.evaluate_run(..)` have an optional flag `abort_on_error` to stop running/evaluating when an error occurs.
- feature: Add `Runner.failed_runs(..)` and `Evaluator.failed_evaluations(..)` to retrieve all failed run / evaluation lineages
- feature: Add `.successful_example_outputs(..)` and `.failed_example_outputs(..)` to `RunRepository` to match the evaluation repository
- feature: Add optional argument to set an id when creating a `Dataset` via `DatasetRepository.create_dataset(..)`
- feature: Traces now log exceptions using the `ErrorValue` type.

- Documentation:
  - feature: Add info on how to run tests in VSCode
  - feature: Add `issue_classification_user_journey` notebook.
  - feature: Add documentation of newly added data retrieval methods `how_to_retrieve_data_for_analysis`
  - feature: Add documentation of release workflow

### Fixes
- fix: Fix version number in pyproject.toml in IL
- fix: Fix instructions for installing IL via pip.

## 0.7.0

### Breaking Changes
- breaking change: FScores are now correctly exposed as FScores and no longer as RougeScores
- breaking change: HuggingFaceAggregationRepository and HuggingFaceDatasetRepository now consistently follow the same folder structure as FileDatasetRepository when creating data sets. This means that datasets will be stored in a folder datasets and additional sub-folders named according to the respective dataset ID.
- breaking change: Split run_repository into file_run_repository, in_memory_run_repository.
- breaking change: Split evaluation_repository into argilla_evaluation_repository, file_evaluation_repository and in_memory_evaluation_repository
- breaking change: Split dataset_repository into file_dataset_repository and in_memory_dataset_respository
- breaking change: Split aggregation_respository into file_aggragation_repository and in_memory_aggregation_repository
- breaking change: Renamed evaluation/run.py to evaluation/run_evaluator.py
- breaking change: Split evaluation/domain and distribute it across aggregation, evaluation, dataset and run packages.
- breaking change: Split evaluation/argilla and distribute it across  aggregation and evaluation packages.
- breaking change: Split evaluation into separate dataset, run, evaluation and aggregationpackages.
- breaking change: Split evaluation/hugging_face.py into dataset and aggregation repository files in data_storage package.
- breaking change: create_dataset now returns the new Dataset type instead of a dataset ID.
- breaking change:  Consistent naming for repository root directories when creating evaluations or aggregations:
  - .../eval → .../evaluations and .../aggregation → aggregations.
- breaking change: Core tasks not longer provide defaults for the applied models.
- breaking change: Methods returning entities from repositories now return the results ordered by their IDs.
- breaking change:  Renamed crashed_during_eval_count to crashed_during_evaluation_count in AggregationOverview.
- breaking change: Renamed create_evaluation_dataset to initialize_evaluation in EvaluationRepository.
- breaking change:  Renamed to_explanation_response  to to_explanation_request in ExplainInput.
- breaking change: Removed TextHighlight::text in favor of TextHighlight::start and TextHighlight::end
- breaking change: Removed `IntelligenceApp` and `IntelligenceStarterApp`
- breaking change: RetrieverBasedQa uses now MultiChunkQa instead of generic task pr SingleChunkQa
- breaking change: EvaluationRepository failed_example_evaluations no longer abstract
- breaking change: Elo calculation simplified:
  - Payoff from elo package has been removed
  - PayoffMatrix from elo package renamed to MatchOutcome
  - SingleChunkQa uses logit_bias to promote not answering for German
- breaking change: Remove ChunkOverlap task.
- breaking change: Rename Chunk to TextChunk.
- breaking change: Rename ChunkTask to Chunk .
- breaking change: Rename EchoTask to Echo.
- breaking change: Rename TextHighlightTask to TextHighlightbreaking change: Rename ChunkOverlaptTask to ChunkOverlap

### New Features
- Aggregation:
  - feature: InstructComparisonArgillaAggregationLogic uses full evaluation set instead of sample for aggregation

- Documentation

  - feature: Added How-To’s (linked in the README):
    - how to define a task
    - how to implement a task
    - how to create a dataset
    - how to run a task on a dataset
    - how to perform aggregation
    - how to evaluate runs
  - feature: Restructured and cleaned up README for more conciseness.
  - feature: Add illustrations to Concepts.md.
  - feature: Added tutorial for adding task to a FastAPI app (linked in README).
  - feature: Improved and added various DocStrings.
  - feature: Added a README section about the client URL.
  - feature: Add python naming convention to README

- Classify
  - feature: PromptBasedClassify now supports changing of the prompt instruction via the instruction parameter.
  - feature: Add default model for PromptBasedClassify
  - feature: Add default task for PromptBasedClassify

- Evaluation
  - feature:  All repositories will return a ValueError when trying to access a dataset that does not exist while also trying to access an entry of the dataset. If only the dataset is retrieved, it will return None.
  - `ArgillaEvaluationRepository` now handles failed evaluations.
  - feature: Added SingleHuggingfaceDatasetRepository.
  - feature: Added HighlightCoverageGrader.
  - feature: Added LanguageMatchesGrader.

  - feature: Added prettier default printing behavior of repository entities by providing overloads to __str__ and __repr__   methods.

  - feature: Added abstract HuggingFace repository base-class.

  - feature: Refactoring of HuggingFace repository

  - feature: Added HuggingFaceAggregationRepository.
  - feature: Added template method to individual repository
  - feature: Added Dataset model to dataset repository. This allows to store a short descriptive name for the dataset for easier identification
  - feature: SingleChunkQa internally now uses the same model in TextHighlight by default.
  - feature: MeanAccumulator tracks standard deviation and standard error
  - feature: EloCalculator now updates ranking after each match
  - feature: Add data selection methods to repositories:
    - AggregationRepository::aggregation_overviews
    - EvaluationRepository::run_overviews
    - EvaluationRepository::run_overview_ids
    - EvaluationRepository::example_output
    - EvaluationRepository::example_outputs
    - EvaluationRepository::example_output_ids
    - EvaluationRepository::example_trace
    - EvaluationRepository::example_tracer
    - RunRepository::run_overviews
    - RunRepository::run_overview_ids
    - RunRepository::example_output
    - RunRepository::example_outputs
    - RunRepository::example_output_ids
    - RunRepository::example_trace
    - RunRepository::example_tracer

  - feature: Evaluator continues in case of no successful outputs

- Q & A

  - feature: Define default parameters for LongContextQa, SingleChunkQa
  - feature: Define default task for RetrieverBasedQa
  - feature: Define default model for KeyWordExtract, MultiChunkQa,
  - feature: Improved focus of highlights in TextHighlight tasks.
  - feature: Added filtering for TextHighlight tasks.
  - feature: Introduce logit_bias to SingleChunkQa

- Summarize
  - feature: Added RecursiveSummarizeInput.
  - feature:  Define defaults for SteerableSingleChunkSummarize,SteerableLongContexSummarize, RecursiveSummarize

- Tracer
  - feature: Added better trace viewer integration:
    - Add trace storage to trace viewer server
    - added submit_to_tracer_viewer method to InMemoryTracer
    - UI and navigation improvements for trace viewer
    - Add exception handling for tracers during log entry writing

- Others

  - feature: The following classes are now exposed:
    - DocumentChunk
    - MultipleChunkQaOutput
    - Subanswer
  - feature: Simplified internal imports.
  - feature: Stream lining of __init__-parameters of all tasks
    - Sub-tasks are typically exposed as `__init__`-parameters with sensible defaults.
    - Defaults for non-trivial parameters like models or tasks are defined in __init__while the default parameter is None.
    - Instead of exposing parameters that are passed on to sub-tasks the sub-task themselves are exposed.
  - feature: Update supported models

### Fixes

- fix: Fixed exception handling in language detection of LanguageMatchesGrader.
- fix: Fixed a bug that could lead to cut-off highlight ranges in TextHighlight tasks.
- fix: Fixed list_ids methods to use path_to_str
- fix: Disallow traces without end in the trace viewer
- fix: ArgillaClient now correctly uses provided API-URL instead of hard-coded localhost

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
