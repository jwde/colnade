# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.2] - 2026-06-27

### Fixed

- **Type-checker false positives on `filter()`, `&`/`|`, and `Schema.Row(...)`** — `Column.__eq__` is now overloaded so `col == value` is typed as `BinOp[Bool]` (accepted by `filter()` and combinable with `&`/`|`) while `col == col` stays a `JoinCondition` for `.join(on=...)`. `Schema.Row(...)` now accepts keyword arguments under static checking instead of resolving to `object()`. Idiomatic Colnade code no longer needs `# ty: ignore` on these (#184)

## [0.8.1] - 2026-03-01

### Fixed

- Examples now use `import colnade as cn` convention, matching tutorials (#170)
- `full_pipeline.py` score column correctly declared nullable (#171)
- Tutorial/example class name mismatch: `ProfileWithStats` → `ProfileWithCounts` (#171)
- Stale "eager only" comments removed for `head()`/`tail()` (#171)
- Comparison table: acknowledge Pandera's Polars support and StaticFrame's `CallGuard` (#173)

### Added

- README: Limitations section documenting type coverage gaps (#172)
- Quick-start: "Enable runtime validation" and "Handling validation errors" sections (#172, #174)
- Installation: backend engine version requirements (#174)
- `SchemaError.null_violations` documented with full attribute table (#174)
- `DataFrame.from_batches()` documented in user guide (#174)
- Dask `from_dict`/`from_rows` documented (#174)

## [0.8.0] - 2026-02-28

### Added

- **`when/then/otherwise` conditional expressions** — build multi-branch if/else logic in the expression DSL: `when(Users.age > 65).then("senior").otherwise("standard")`. Supports chained conditions and works with all three backends (#159)
- **`concat()` for vertical concatenation** — stack DataFrames or LazyFrames vertically with type-safe schema preservation: `concat(df1, df2, df3)`. Validates all inputs share the same schema (#160)
- **`.item()` for scalar extraction** — extract a Python scalar from a 1-row DataFrame or LazyFrame: `df.agg(...).item()` or `df.head(1).item(Users.name)`. Return type is inferred from the column dtype via overloads (#161)

### Fixed

- **`.item()` on Polars LazyFrame** no longer crashes — was missing `.collect()` before accessing `.shape` (#165)
- **`.item()` on Pandas/Dask** now returns `None` instead of leaking `pd.NA` for null cells (#165)

### Changed

- Documentation adopts `import colnade as cn` convention across all examples (#166)

### Infrastructure

- README images use absolute URLs for PyPI rendering (#164)
- Performance docs rewritten with accurate benchmarks (#151)
- Three rounds of documentation audit fixes (#153, #155, #157)
- Removed spec.md — superseded by docs/ and CLAUDE.md (#152)

## [0.7.0] - 2026-02-24

### Removed

- **`UntypedDataFrame` and `UntypedLazyFrame`** — these string-based escape hatch types have been removed. Use `with_raw()` for scoped engine-native operations or `to_native()` for full escape (#150)

### Added

- **Schema inheritance in `cast_schema`** — `with_columns` + `cast_schema` to a child schema now works. When the target schema extends the source schema, unresolved columns fall back to identity mapping, enabling the common pattern of adding computed columns via `with_columns` and binding to a richer child schema (#150)

### Fixed

- **`str_contains` uses literal matching** — Polars backend now correctly uses `literal=True` instead of treating the argument as a regex pattern (#130, #148)
- **`assert_non_null` actually raises** — now raises `SchemaError` when null values are present instead of being a no-op (#129, #147)
- **`__and__`, `__or__`, `__invert__` return types** — boolean operators now correctly return `BinOp[Bool]` / `UnaryOp[Bool]` instead of `BinOp[Any]` (#137, #146)
- Various documentation fixes and example code cleanup (#120, #121, #123, #124, #125, #128, #132, #133, #134, #135, #136)

### Infrastructure

- Switched from Codecov to coverage-comment-action (#119)

## [0.6.0] - 2026-02-23

### Added

- **Self-narrowing for dtype-restricted methods** — calling `.sum()` on a string column, `.str_contains()` on a numeric column, `.is_nan()` on an integer column, etc. is now a static type error in ty, mypy, and pyright. 22 methods across numeric, float, string, temporal, struct, and list categories are restricted to appropriate dtypes (#115, #116)

### Fixed

- **`over()` window function in Pandas and Dask** — `col.sum().over(partition)` now correctly computes per-group aggregations instead of returning raw column values. Polars was unaffected (#112, #117)

## [0.5.5] - 2026-02-22

### Fixed

- **`unique()` with no columns** — all three backends now correctly deduplicate across all columns when no subset is specified (#108, #109)
- **Missing aggregations on Pandas and Dask** — `std()`, `var()`, and `n_unique()` now work in Pandas and Dask backends (#110, #111)

### Added

- **`LazyFrame.height` and `len()`** — trigger computation on lazy backends (#113)
- **`LazyFrame.to_batches()`** — converts to Arrow batches, triggering computation on lazy backends (#114)

## [0.5.4] - 2026-02-22

### Fixed

- **Restore `from_dict` and `from_rows` to Dask backend** — these construct DataFrames from in-memory data and were incorrectly removed in v0.5.3. They now return `LazyFrame` (not `DataFrame`) to match Dask's lazy semantics (#107)

## [0.5.3] - 2026-02-22

### Changed

- **Dask backend: remove eager I/O** — `read_parquet`, `read_csv` removed from `colnade-dask`. Dask is inherently lazy, so only `scan_parquet` and `scan_csv` are now provided (#105)

### Added

- **`LazyFrame.head()` and `LazyFrame.tail()`** — added for parity with `DataFrame` (#105)

## [0.5.2] - 2026-02-22

### Fixed

- **Dtype validation with NumPy types** — `read_parquet` with structural validation no longer rejects valid data when Dask/Pandas returns NumPy dtypes (`uint64`, `float64`) instead of Pandas extension types (`pd.UInt64Dtype()`, `pd.Float64Dtype()`) (#102, #103)

## [0.5.1] - 2026-02-22

### Added

- **Ungrouped aggregation** — `DataFrame.agg()` and `LazyFrame.agg()` to aggregate all rows into a single output row without `group_by` (#97, #98)

### Fixed

- **Dask list operations** — added `meta=` to all `.apply()` calls in `_translate_list_op` to fix metadata inference failures (#96)
- **Dask `is_not_null`** — fixed to use `.notnull()` instead of `.notna()` which Dask doesn't support (#96)
- **Dask list `mean`** — fixed boolean evaluation of Arrow arrays in mean lambda (#96)

### Changed

- **CI coverage threshold** raised from 80% to 95% (#96)

## [0.5.0] - 2026-02-22

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

[Unreleased]: https://github.com/jwde/colnade/compare/v0.8.2...HEAD
[0.8.2]: https://github.com/jwde/colnade/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/jwde/colnade/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/jwde/colnade/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/jwde/colnade/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jwde/colnade/compare/v0.5.5...v0.6.0
[0.5.5]: https://github.com/jwde/colnade/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/jwde/colnade/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/jwde/colnade/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/jwde/colnade/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/jwde/colnade/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/jwde/colnade/releases/tag/v0.5.0
