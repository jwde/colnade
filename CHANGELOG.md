# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.5.0]: https://github.com/jwde/colnade/releases/tag/v0.5.0
