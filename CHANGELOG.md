# Changelog

## Unreleased
- feature: `AlephAlphaModel` now supports a `context_size`-property

### Breaking Changes
...

### New Features
...

### Fixes
...

### Deprecations
...

## 1.0.0

Initial stable release

With the release of version 1.0.0 there have been introduced some new features but also some breaking changes you should be aware of.
Apart from these changes, we also had to reset our commit history, so please be aware of this fact.

### Breaking Changes
- breaking_change: The TraceViewer has been exported to its own repository and can be accessed via the artifactory [here]( https://alephalpha.jfrog.io.)
- breaking_change: `HuggingFaceDatasetRepository` now has a parameter caching, which caches  examples of a dataset once loaded.
  - `True` as default value
  - set to `False` for **non-breaking**-change


### New Features
#### Llama2 and LLama3 model support
- feature: Introduction of `LLama2InstructModel` allows support of the LLama2-models:
  - `llama-2-7b-chat`
  - `llama-2-13b-chat`
  - `llama-2-70b-chat`
- feature: Introduction of `LLama3InstructModel` allows support of the LLama2-models:
  - `llama-3-8b-instruct`
  - `llama-3-70b-instruct`
#### DocumentIndexClient
`DocumentIndexClient` has been enhanced with the following set of features:
- feature: `create_index`
- feature `index_configuration`
- feature: `assign_index_to_collection`
- feature: `delete_index_from_collection`
- feature: `list_assigned_index_names`

#### Miscellaneous
- feature: `ExpandChunks`-task now caches chunked documents by ID
- feature: `DocumentIndexRetriever` now supports `index_name`
- feature: `Runner.run_dataset` now has a configurable number of workers via `max_workers` and defaults to the previous value, which is 10.
- feature: In case a `BusyError` is raised during a `complete` the `LimitedConcurrencyClient` will retry until `max_retry_time` is reached.

### Fixes
- fix: `HuggingFaceRepository` no longer is a dataset repository. This also means that `HuggingFaceAggregationRepository` no longer is a dataset repository.
- fix: The input parameter of the `DocumentIndex.search()`-function now has been renamed from `index` to `index_name`
