# API Reference

## Core Modules

| Module | Description |
|--------|-------------|
| [`colnade.schema`](schema.md) | Schema base class, Column descriptors, mapped_from, SchemaError |
| [`colnade.dataframe`](dataframe.md) | DataFrame, LazyFrame, GroupBy, JoinedDataFrame, Untyped frames |
| [`colnade.expr`](expressions.md) | Expression AST nodes (ColumnRef, BinOp, Agg, etc.) |
| [`colnade.dtypes`](types.md) | Data type definitions (UInt64, Float64, Utf8, Struct, List, etc.) |

## Backend Adapters

| Module | Description |
|--------|-------------|
| [`colnade_polars`](polars.md) | Polars backend: PolarsBackend, I/O functions, dtype mapping |
