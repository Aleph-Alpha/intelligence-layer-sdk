# Changelog
## Unreleased
...
### Features
 - Introduced `AsyncDocumentIndexClient` and `AsyncDocumentIndexRetriever` as drop-in replacements for their blocking counterparts, enabling coroutine-based, non-blocking document indexing and retrieval.
### Fixes
- `InMemoryDatasetRepository` now has a more descriptive error message when creating a dataset fails due to an ID clash.
- `StudioClient` now deserializes and serializes examples while maintaining type information, which was previously dropped.
- `RunRepository` and `EvaluationRepository` now more accurately reflect their actual return types in their signatures. Previously, it was not obvious that failed examples could be returned.

### Deprecations
...

### Breaking Changes
-  `InMemoryDatasetRepository`, `InMemoryRunRepository`, `InMemoryEvaluationRepository`, and `InMemoryAggregationRepository` now either return the exact types given by users when retrieving example-related data or fail. Specifically, this means that passing the wrong type when retrieving data will now fail with a `ValidationError`. Previously, the repositories disregarded the types they were given and returned whatever object was saved.
  -  This is in line with how the other repositories work.
-  `EloQaEvaluationLogic` now has an expected output type of `None` instead of `SingleChunkQaOutput`. The information was unused.
  -  If you have pipelines that define data to be processed by this logic, or if you subclass from this specific logic, you may need to adapt it.
  - `log_probs` have been set to 20 instead of the prior value of 30
- The legacy `Trace Viewer` has now been removed along with all calls to it.

## 9.1.0
...
### Features
- New Pharia Kernel connector (`KernelTask`) for calling Skills from a Task
- Add `HybridQdrantInMemoryRetriever` enabling hybrid search for in-memory Qdrant collections

### Fixes
 - Add warning to `PromptBasedClassify` and `PromptBasedClassifyWithDefinitions` to be cautious when using them with model families other than luminous

### Deprecations
...
### Breaking Changes
...

## 9.0.2
### Features
- Bump dependency versions

## 9.0.1
### Fixes
 - Fixes an incompatibility where models with tokenizer with no whitespace prefix could not be used for qa examples. Now, no error will be thrown.

## 9.0.0

### Features
- Introduce `Benchmark` and `StudioBenchmark`
  - `Benchmark` allows you to evaluate and compare the performance of different `Task`s  with a fixed evaluation logic, aggregation logic and `Dataset`.
  - Add `how_to_execute_a_benchmark.ipynb` to how-tos
  - Add `studio.ipynb` to notebooks to show how one can debug a `Task` with Studio
- Introduce `BenchmarkRepository`and `StudioBenchmarkRepository`
- Add `create_project` bool to `StudioClient.__init__()` to enable users to automatically create their Studio projects
- Add progressbar to the `Runner` to be able to track the `Run`
- Add `StudioClient.submit_benchmark_lineages` function and include it in `StudioClient.submit_benchmark_execution`

#### DocumentIndexClient 
- Add method `DocumentIndexClient.chunks()` for retrieving all text chunks of a document.
- Add metadata filter `FilterOps.IS_NULL`, that allows to filter fields based on whether their value is null.

### Fixes
- The Document Index `SearchQuery` now correctly allows searches with a negative `min_score`.

### Deprecations
...

### Breaking Changes
 - The env variable `POSTGRES_HOST` is split into `POSTGRES_HOST` and `POSTGRES_PORT`. This affects all classes interacting with Studio and the `InstructionFinetuningDataRepository`.
 - The following env variables now need to be set (previously pointed to defaults)
   - `CLIENT_URL` - URL of your inference stack
   - `DOCUMENT_INDEX_URL` - URL of the document index

## 8.0.0

### Features
- You can now customise the embedding model when creating an index using the `DocumentIndexClient`.
- You can now use the `InstructableEmbed` embedding strategy when creating an index using the `DocumentIndexClient`. See the `document_index.ipynb` notebook for more information and an example.

### Breaking Changes
- The way you configure indexes in the `DocumentIndexClient` has changed. See the `document_index.ipynb` notebook for more information.
    - The `EmbeddingType` alias has been renamed to `Representation` to better align with the underlying API.
    - The `embedding_type` field has been removed from the `IndexConfiguration` class. You now configure embedding-related parameters via the `embedding` field.
    - You now always need to specify an embedding model when creating an index. Previously, this was always `luminous-base`.

## 7.3.1
### Features
- Dependency updates

## 7.3.0

### Features
- Add support for Llama3InstructModel in PromptBasedClassify
- Add TextControl to 'to_instruct_prompt' for instruct models
  - Add 'attention_manipulation_with_text_controls.ipynb' to tutorial notebooks
- Introduced `InstructionFinetuningDataHandler` to provide methods for storing, retrieving and updating finetuning data samples given an `InstructionFinetuningDataRepository`. Also has methods for filtered sample retrieval and for dataset formatting.
- Introduced `InstructionFinetuningDataRepository` for storing and retrieving finetuning samples. Comes in two implementations:
  - `PostgresInstructionFinetuningDataRepository` to work with data stored in a Postgres database.
  - `FileInstructionFinetuningDataRepository` to work with data stored in the local file-system.
- Compute precision, recall and f1-score by class in `SingleLabelClassifyAggregationLogic`
- Add submit_dataset function to StudioClient
  - Add `how_to_upload_existing_datasets_to_studio.ipynb` to how-tos

### Fixes
 - Improved some docstring inconsistencies across the codebase and switched the docstring checker to pydoclint.

## 7.2.0

### Features
- Add support for stages and files in Data client.
- Add more in-depth description for `MiltipleChunRetrieverQaOutput` and `ExpandChunks`

### Fixes
- Data repository media types now validated with a function instead of an Enum.
- Update names of `pharia-1` models to lowercase, aligning with fresh deployments of the api-scheduler.

## 7.1.0

### Features
- Add Catalan and Polish support to `DetectLanguage`.
- Add utility function `run_is_already_computed` to `Runner` to check if a run with the given metadata has already been computed.
  - The `parameter_optimization` notebook describes how to use the `run_is_already_computed` function.
  
### Fixes
- The default `max_retry_time` for the `LimitedConcurrencyClient` is now set to 3 minutes from a day. If you have long-running evaluations that need this, you can re-set a long retry time in the constructor.


## 7.0.0

### Features
- You can now specify a `hybrid_index` when creating an index for the document index to use hybrid (semantic and keyword) search.
- `min_score` and `max_results` are now optional parameters in `DocumentIndexClient.SearchQuery`.
- `k` is now an optional parameter in `DocumentIndexRetriever`.
- List all indexes of a namespace with `DocumentIndexClient.list_indexes`.
- Remove an index from a namespace with `DocumentIndexClient.delete_index`.
- `ChatModel` now inherits from `ControlModel`. Although we recommend to use the new chat interface, you can use the `Pharia1ChatModel` with tasks that rely on `ControlModel` now.

### Fixes
- `DocumentIndexClient` now properly sets `chunk_overlap` when creating an index configuration.

### Breaking Changes
- The default model for `Llama3InstructModel` is now `llama-3.1-8b-instruct` instead of `llama-3-8b-instruct`. We also removed the llama3.0 models from the recommended models of the `Llama3InstructModel`.
- The default value of `threshold` in the `DocumentIndexRetriever` has changed from `0.5` to `0.0`. This accommodates fusion scoring for searches over hybrid indexes.


## 6.0.0

### Features
- Remove cap for `max_concurrency` in `LimitedConcurrencyClient`.
- Introduce abstract `LanguageModel` class to integrate with LLMs from any API
  - Every `LanguageModel` supports echo to retrieve log probs for an expected completion given a prompt
- Introduce abstract `ChatModel` class to integrate with chat models from any API
  - Introducing `Pharia1ChatModel` for usage with pharia-1 models.
  - Introducing `Llama3ChatModel` for usage with llama models.
- Upgrade `ArgillaWrapperClient` to use Argilla v2.x
- (Beta) Add `DataClient` and `StudioDatasetRepository` as connectors to Studio for submitting data.
- Add the optional argument `generate_highlights` to `MultiChunkQa`, `RetrieverBasedQa` and `SingleChunkQa`. This makes it possible to disable highlighting for performance reasons.

### Fixes
- Increase number of returned `log_probs` in `EloQaEvaluationLogic` to avoid missing a valid answer

### Deprecations 
- Removed `DefaultArgillaClient`
- Deprecated `Llama2InstructModel` 

### Breaking Changes
- We needed to upgrade argilla-server image version from `argilla-server:v1.26.0` to `argilla-server:v1.29.0` to maintain compatibility.
  - Note: We also updated our elasticsearch argilla backend to `8.12.2`


## 5.1.0

### Features
- Updated `DocumentIndexClient` with support for metadata filters.
    - Add documentation for filtering to `document_index.ipynb`.
- Add `StudioClient` as a connector for submitting traces.
- You can now specify a `chunk_overlap` when creating an index in the Document Index.
- Add support for monitoring progress in the document index connector when embedding documents.

### Fixes
 - TaskSpan now properly sets its status to `Error` on crash.

### Deprecations 
 - Deprecate old Trace Viewer as the new `StudioClient` replaces it. This affects `Tracer.submit_to_trace_viewer`.

## 5.0.3

### Fixes
- Update docstrings for 'calculate_bleu' in 'BleuGrader' to now correctly reflect float range from 0 to 100 for the return value.

## 5.0.2

### Fixes
- Reverted a bug introduced in `MultipleChunkRetrieverQa` text highlighting.


## 5.0.1

### Fixes
- Serialization and deserialization of `ExportedSpan` and its `attributes` now works as expected.
- `PromptTemplate.to_rich_prompt` now always returns an empty list for prompt ranges that are empty.
- `SingleChunkQa` no longer crashes if given an empty input and a specific prompt template. This did not affect users who used models provided in `core`.
- Added default values for `labels` and `metadata` for `EvaluationOverview` and `RunOverview`
- In the `MultipleChunkRetrieverQa`, text-highlight start and end points are now restricted to within the text length of the respective chunk.


## 5.0.0

### Breaking Changes
- `RunRepository.example_output`  now returns `None` and prints a warning when there is no associated record for the given `run_id` instead of raising a `ValueError`.
- `RunRepository.example_outputs` now returns an empty list and prints a warning when there is no associated record for the given `run_id` instead of raising a `ValueError`.

### Features
 -  `Runner.run_dataset` can now be resumed after failure by setting the `resume_from_recovery_data` flag to `True` and calling `Runner.run_dataset` again.
   - For `InMemoryRunRepository` based `Runner`s this is limited to runs that failed with an exception that did not crash the whole process/kernel.
   - For `FileRunRepository` based `Runners` even runs that crashed the whole process can be resumed.
   - `DatasetRepository.examples` now accepts an optional parameter `examples_to_skip` to enable skipping of `Example`s with the provided IDs.
   - Add `how_to_resume_a_run_after_a_crash` notebook.

### Fixes
 - Remove unnecessary dependencies from IL
 - Added default values for `labels` and `metadata` for `PartialEvaluationOverview`


## 4.1.0

### New Features
  - Add `eot_token` property to `ControlModel` and derived classes (`LuminousControlModel`, `Llama2InstructModel` and `Llama3InstructModel`) and let `PromptBasedClassify` use this property instead of a hardcoded string.
  - Introduce a new argilla client `ArgillaWrapperClient`. This uses the `argilla` package as a connection to argilla and supports all question types that argilla supports in their `FeedbackDataset`. This includes text and yes/no questions. For more information about the questions, check [their official documentation](https://docs.argilla.io/en/latest/practical_guides/create_update_dataset/create_dataset.html#define-questions).
    - Changes to switch: 
      - `DefaultArgillaClient` -> `ArgillaWrapperClient`
      - `Question` -> `argilla.RatingQuestion`, `options` -> `values` and it takes only a list
      - `Field` -> `argilla.TextField`
  - Add `description` parameter to `Aggregator.aggregate_evaluation` to allow individual descriptions without the need to create a new `Aggregator`. This was missing from the previous release.
  - Add optional field `metadata` to `Dataset`, `RunOverview`, `EvaluationOverview` and `AggregationOverview`
    - Update `parameter_optimization.ipynb` to demonstrate usage of metadata****
  - Add optional field `label` to `Dataset`, `RunOverview`, `EvaluationOverview` and `AggregationOverview`
  - Add `unwrap_metadata` flag to `aggregation_overviews_to_pandas` to enable inclusion of metadata in pandas export. Defaults to True.

### Fixes
  - Reinitializing different `AlephAlphaModel` instances and retrieving their tokenizer should now consume a lot less memory.
  - Evaluations now raise errors if ids of examples and outputs no longer match. If this happens, continuing the evaluation would only produce incorrect results.
  - Performing evaluations on runs with a different number of outputs now raises errors. Continuing the evaluation in this case would only lead to an inconsistent state.

## 4.0.1

### Breaking Changes
 - Remove the `Trace` class, as it was no longer used.
 - Renamed `example_trace` to `example_tracer` and changed return type to `Optional[Tracer]`.
 - Renamed `example_tracer` to `create_tracer_for_example`.
 - Replaced langdetect with lingua as language detection tool. This mean that old thresholds for detection might need to be adapted.

### New Features
 - `Lineages` now contain `Tracer` for individual `Output`s.
 - `convert_to_pandas_data_frame` now also creates a column containing the `Tracer`s.
 - `run_dataset` now has a flag `trace_examples_individually` to create `Tracer`s for each example. Defaults to True.
 - Added optional `metadata` field to `Example`.

### Fixes
  - ControlModels throw a warning instead of an error in case a not-recommended model is selected.
  - The `LimitedConcurrencyClient.max_concurrency` is now capped at 10, which is its default, as the underlying `aleph_alpha_client` does not support more currently.
  - ExpandChunk now works properly if the chunk of interest is not at the beginning of a very large document. As a consequence, `MultipleChunkRetrieverQa` now works better with larger documents and should return fewer `None` answers.


## 3.0.0

### Breaking Changes
 - We removed the `trace_id` as a concept from various tracing-related functions and moved them to a `context`. If you did not directly use the `trace_id` there is nothing to change.
   - `Task.run` no longer takes a trace id. This was a largely unused feature, and we revamped the trace ids for the traces.
   - Creating `Span`, `TaskSpan` or logs no longer takes `trace_id`. This is handled by the spans themselves, who now have a `context` that identifies them.
     - `Span.id` is therefore also removed. This can be accessed by `span.context.trace_id`, but has a different type.
   - The `OpenTelemetryTracer` no longer logs a custom `trace_id` into the attributes. Use the existing ids from its context instead.
   - Accessing a single trace from a `PersistentTracer.trace()` is no longer supported, as the user does not have access to the `trace_id` anyway. The function is now called `traces` and returns all available traces for a tracer.
 - `InMemoryTracer` and derivatives are no longer `pydantic.BaseModel`. Use the `export_for_viewing` function to export a serializable representation of the trace.
 - We updated the graders to support python 3.12 and moved away from `nltk`-package:
    - `BleuGrader` now uses `sacrebleu`-package.
    - `RougeGrader` now uses the `rouge_score`-package.
 - When using the `ArgillaEvaluator`, attempting to submit to a dataset, which already exists, will no longer work append to the dataset. This makes it more in-line with other evaluation concepts.
   - Instead of appending to an active argilla dataset, you now need to create a new dataset, retrieve it and then finally combine both datasets in the aggregation step.
   - The `ArgillaClient` now has methods `create_dataset` for less fault-ignoring dataset creation and `add_records` for performant uploads.

### New Features
 - Add support for Python 3.12
 - Add `skip_example_on_any_failure` flag to `evaluate_runs` (defaults to True). This allows to configure if you want to keep an example for evaluation, even if it failed for some run.
 - Add `how_to_implement_incremental_evaluation`.
 - Add `export_for_viewing` to tracers to be able to export traces in a unified format similar to OpenTelemetry.
   - This is not supported for the `OpenTelemetryTracer` because of technical incompatibilities.
 - All exported spans now contain the status of the span.
 - Add `description` parameter to `Evaluator.evaluate_runs` and `Runner.run_dataset` to allow individual descriptions without the need to create a new `Evaluator` or `Runner`.
 - All models raise an error during initialization if an incompatible `name` is passed, instead of only when they are used.
 - Add `aggregation_overviews_to_pandas` function to allow for easier comparison of multiple aggregation overviews.
 - Add `parameter_optimization.ipynb` notebook to demonstrate the optimization of tasks by comparing different parameter combinations.
 - Add `convert_file_for_viewing` in the `FileTracer` to convert the trace file format to the new (OpenTelemetry style) format and save as a new file.
 - All tracers can now call `submit_to_trace_viewer` to send the trace to the Trace Viewer.

### Fixes
 - The document index client now correctly URL-encodes document names in its queries.
 - The `ArgillaEvaluator` not properly supports `dataset_name`.
 - Update outdated `how_to_human_evaluation_via_argilla.ipynb`.
 - Fix bug in `FileSystemBasedRepository` causing spurious mkdir failure if the file actually exists.
 - Update broken README links to Read The Docs.
 - Fix a broken multi-label classify example in the `evaluation` tutorial.

## 2.0.0

### Breaking Changes
 - Changed the behavior of `IncrementalEvaluator::do_evaluate` such that it now sends all `SuccessfulExampleOutput`s to `do_incremental_evaluate` instead of only the new `SuccessfulExampleOutput`s.

### New Features
 - Add generic `EloEvaluationLogic` class for implementation of Elo evaluation use cases.
 - Add `EloQaEvaluationLogic` for Elo evaluation of QA runs, with optional later addition of more runs to an existing evaluation.
 - Add `EloAggregationAdapter` class to simplify using the `ComparisonEvaluationAggregationLogic` for different Elo use cases.
 - Add `elo_qa_eval` tutorial notebook describing the use of an (incremental) Elo evaluation use case for QA models.
 - Add `how_to_implement_elo_evaluations` how-to as skeleton for implementing Elo evaluation cases

### Fixes
- `ExpandChunks`-task is now fast even for very large documents

## 1.2.0

We did a major revamp of the `ArgillaEvaluator` to separate an `AsyncEvaluator` from the normal evaluation scenario.
This comes with easier to understand interfaces, more information in the `EvaluationOverview` and a simplified aggregation step for Argilla that is no longer dependent on specific Argilla types.
Check the how-to for detailed information [here](./src/documentation/how_tos/how_to_human_evaluation_via_argilla.ipynb)

### Breaking Changes

- rename: `AggregatedInstructComparison` to `AggregatedComparison`
- rename `InstructComparisonArgillaAggregationLogic` to `ComparisonAggregationLogic`
- remove: `ArgillaAggregator` - the regular aggregator now does the job
- remove: `ArgillaEvaluationRepository` - `ArgillaEvaluator` now uses `AsyncRepository` which extend existing `EvaluationRepository` for the human-feedback use-case
- `ArgillaEvaluationLogic` now uses `to_record` and `from_record` instead of `do_evaluate`. The signature of the `to_record` stays the same. The `Field` and `Question` are now defined in the logic instead of passed to the `ArgillaRepository`
- `ArgillaEvaluator` now takes the `ArgillaClient` as well as the `workspace_id`. It inherits from the abstract `AsyncEvaluator` and no longer has `evalaute_runs` and `evaluate`. Instead it has `submit` and `retrieve`.
- `EvaluationOverview` gets attributes `end_date`, `successful_evaluation_count` and `failed_evaluation_count`
  - rename: `start` is now called `start_date` and no longer optional
- we refactored the internals of `Evaluator`. This is only relevant if you subclass from it. Most of the typing and data handling is moved to `EvaluatorBase`

### New Features
- Add `ComparisonEvaluation` for the elo evaluation to abstract from the Argilla record
- Add `AsyncEvaluator` for human-feedback evaluation. `ArgillaEvaluator` inherits from this
  - `.submit` pushes all evaluations to Argilla to label them
  - Add `PartialEvaluationOverview` to store the submission details.
  - `.retrieve` then collects all labelled records from Argilla and stores them in an `AsyncRepository`.
  - Add `AsyncEvaluationRepository` to store and retrieve `PartialEvaluationOverview`. Also added `AsyncFileEvaluationRepository` and `AsyncInMemoryEvaluationRepository`
- Add `EvaluatorBase` and `EvaluationLogicBase` for base classes for both async and synchronous evaluation.

### Fixes
 - Improve description of using artifactory tokens for installation of IL
 - Change `confusion_matrix` in `SingleLabelClassifyAggregationLogic` such that it can be persisted in a file repository

## 1.1.0

### New Features
 - `AlephAlphaModel` now supports a `context_size`-property
 - Add new `IncrementalEvaluator` for easier addition of runs to existing evaluations without repeated evaluation.
   - Add `IncrementalEvaluationLogic` for use in `IncrementalEvaluator`

## 1.0.0

Initial stable release

With the release of version 1.0.0 there have been introduced some new features but also some breaking changes you should be aware of.
Apart from these changes, we also had to reset our commit history, so please be aware of this fact.

### Breaking Changes
-  The TraceViewer has been exported to its own repository and can be accessed via the artifactory [here]( https://alephalpha.jfrog.io.)
-  `HuggingFaceDatasetRepository` now has a parameter caching, which caches  examples of a dataset once loaded.
  - `True` as default value
  - set to `False` for **non-breaking**-change


### New Features
#### Llama2 and LLama3 model support
-  Introduction of `LLama2InstructModel` allows support of the LLama2-models:
  - `llama-2-7b-chat`
  - `llama-2-13b-chat`
  - `llama-2-70b-chat`
-  Introduction of `LLama3InstructModel` allows support of the LLama2-models:
  - `llama-3-8b-instruct`
  - `llama-3-70b-instruct`
#### DocumentIndexClient
`DocumentIndexClient` has been enhanced with the following set of features:
-  `create_index`
- feature `index_configuration`
-  `assign_index_to_collection`
-  `delete_index_from_collection`
-  `list_assigned_index_names`

#### Miscellaneous
-  `ExpandChunks`-task now caches chunked documents by ID
-  `DocumentIndexRetriever` now supports `index_name`
-  `Runner.run_dataset` now has a configurable number of workers via `max_workers` and defaults to the previous value, which is 10.
-  In case a `BusyError` is raised during a `complete` the `LimitedConcurrencyClient` will retry until `max_retry_time` is reached.

### Fixes
-  `HuggingFaceRepository` no longer is a dataset repository. This also means that `HuggingFaceAggregationRepository` no longer is a dataset repository.
-  The input parameter of the `DocumentIndex.search()`-function now has been renamed from `index` to `index_name`
