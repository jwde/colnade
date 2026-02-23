# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.4] - 2026-02-23

### Fixed

- **Restore `from_dict` and `from_rows` to Dask backend** — these construct DataFrames from in-memory data and were incorrectly removed in v0.5.3. They now return `LazyFrame` (not `DataFrame`) to match Dask's lazy semantics (#107)

## [0.5.3] - 2026-02-23

### Changed

- **Dask backend: remove eager I/O** — `read_parquet`, `read_csv` removed from `colnade-dask`. Dask is inherently lazy, so only `scan_parquet` and `scan_csv` are now provided (#105)

### Added

- **`LazyFrame.head()` and `LazyFrame.tail()`** — added for parity with `DataFrame` (#105)

## [0.5.2] - 2026-02-22

### Fixed

- **Dtype validation with NumPy types** — `read_parquet` with structural validation no longer rejects valid data when Dask/Pandas returns NumPy dtypes (`uint64`, `float64`) instead of Pandas extension types (`pd.UInt64Dtype()`, `pd.Float64Dtype()`) (#102, #103)

## [0.5.1] - 2026-02-11

### Added

- **Ungrouped aggregation** — `DataFrame.agg()` and `LazyFrame.agg()` to aggregate all rows into a single output row without `group_by` (#97, #98)

### Fixed

- **Dask list operations** — added `meta=` to all `.apply()` calls in `_translate_list_op` to fix metadata inference failures (#96)
- **Dask `is_not_null`** — fixed to use `.notnull()` instead of `.notna()` which Dask doesn't support (#96)
- **Dask list `mean`** — fixed boolean evaluation of Arrow arrays in mean lambda (#96)

### Changed

- **CI coverage threshold** raised from 80% to 95% (#96)

## [0.5.0] - 2026-02-11

First public release.

### Added

- **Core type system** — `Schema`, `Column[DType]`, typed expressions (`Expr[Bool]`, `Expr[Float64]`, etc.)
- **DataFrame[S] and LazyFrame[S]** — schema-parameterized wrappers preserving type through filter, sort, with_columns, head, tail, limit, drop_nulls
- **Expression DSL** — arithmetic, comparison, logical, string, and null-handling operators on typed column descriptors
- **Nested types** — `Struct[S]` field access and `List[T]` accessor (len, get, contains, sum, mean, min, max)
- **Join system** — `JoinedDataFrame[L, R]` with typed `JoinCondition` from cross-schema `==`
- **cast_schema / mapped_from** — flatten joined data or rename columns with compile-time checked mappings
- **Schema-polymorphic generics** — `DataFrame[S]` with `TypeVar` bound to `Schema` for reusable utility functions
- **Value-level constraints** — `Field(ge=, le=, gt=, lt=, min_length=, max_length=, pattern=, unique=, isin=)` and `@schema_check` for cross-column invariants
- **Three validation levels** — `OFF`, `STRUCTURAL` (columns + dtypes + nullability), `FULL` (structural + value constraints)
- **Typed construction** — `from_rows(Schema, [Schema.Row(...)])` and `from_dict(Schema, {...})` with `Row[S]` generic base class
- **Runtime cross-schema guard** — expressions referencing wrong schema raise at runtime when validation is enabled
- **Backend adapters** — Polars, Pandas, and Dask with expression translation, validation, and I/O
- **Lazy execution** — `scan_parquet`, `scan_csv` with Polars LazyFrame backend
- **I/O** — `read_parquet`, `read_csv`, `write_parquet`, `write_csv` for all backends
- **Untyped escape hatch** — `df.untyped()` drops to string-based columns, `untyped.to_typed(Schema)` re-binds
- **Documentation site** — tutorials, user guide, API reference, comparison page at colnade.com

[Unreleased]: https://github.com/jwde/colnade/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/jwde/colnade/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/jwde/colnade/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/jwde/colnade/releases/tag/v0.5.0
