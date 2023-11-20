# Changelog

## Next version:

### Breaking Changes

- `Dataset` is now a protocol. `SequenceDataset` replaces the old `Dataset`.
- The `ident` attribute on `Example` is now `id`.

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
