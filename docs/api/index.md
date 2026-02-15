# API Reference

## Core Modules

| Module | Description |
|--------|-------------|
| [`colnade core`](schema.md) | Schema, Column, BackendProtocol, ArrowBatch, validation utilities |
| [`colnade.dataframe`](dataframe.md) | DataFrame, LazyFrame, GroupBy, JoinedDataFrame, Untyped frames |
| [`colnade.expr`](expressions.md) | Expression AST nodes (ColumnRef, BinOp, Agg, etc.) |
| [`colnade.dtypes`](types.md) | Data type definitions (UInt64, Float64, Utf8, Struct, List, etc.) |

## Backend Adapters

| Module | Description |
|--------|-------------|
| [`colnade_polars`](polars.md) | Polars backend: PolarsBackend, I/O functions, dtype mapping |
| [`colnade_pandas`](pandas.md) | Pandas backend: PandasBackend, I/O functions, dtype mapping |
| [`colnade_dask`](dask.md) | Dask backend: DaskBackend, I/O functions, scan functions |
