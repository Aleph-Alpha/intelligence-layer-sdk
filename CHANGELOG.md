# Changelog

## Unreleased

### Breaking Changes
...

### New Features
...

### Fixes
...

### Deprecations
...

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
