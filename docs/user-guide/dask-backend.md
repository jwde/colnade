# Dask Backend

The Dask backend (`colnade-dask`) lets you use Colnade's typed DataFrame API on Dask's distributed DataFrames. This page covers Dask-specific semantics that differ from the Polars and Pandas backends.

## Dask is inherently lazy

Unlike Polars (which has separate eager `DataFrame` and lazy `LazyFrame`), Dask DataFrames are always lazy — operations build a task graph that executes only when you call `.compute()`.

Because Dask is always lazy, the Dask backend only provides `scan_parquet` and `scan_csv` (returning `LazyFrame`). There are no eager `read_parquet`/`read_csv` functions — use `scan_*` followed by `.collect()` when you need materialized results.

```python
from colnade_dask import scan_parquet

lf = scan_parquet("users.parquet", Users)   # LazyFrame — builds task graph
filtered = lf.filter(Users.age > 25)        # Still lazy
result = filtered.collect()                 # Triggers computation → DataFrame
```

## Which operations trigger computation

Most operations are deferred. A few must materialize data:

| Operation | Triggers Compute | Notes |
|-----------|:---:|-------|
| `filter`, `sort`, `limit` | No | Builds task graph |
| `head(n)`, `tail(n)` | No | Builds task graph |
| `unique`, `drop_nulls` | No | Builds task graph |
| `with_columns`, `select` | No | Builds task graph |
| `group_by().agg()` | No | Builds task graph |
| `join` | No | Builds task graph |
| `cast_schema` | No | Lazy rename + column selection |
| `collect()` | **Yes** | Explicit materialization |
| `len()` / `height` | **Yes** | Requires counting rows across partitions |
| `iter_rows_as()` | **Yes** | Must materialize to iterate |
| `to_batches()` | **Yes** | Computes each partition |

## Validation and Dask

### Structural validation (`STRUCTURAL`)

Structural validation checks column names, dtypes, and nullability. With Dask:

- **Column names and dtypes** are available without computation — they're part of Dask's metadata.
- **Null checks** require computation — Dask must scan data to determine if non-nullable columns contain nulls.

```python
import colnade as cn
from colnade_dask import scan_parquet

cn.set_validation(cn.ValidationLevel.STRUCTURAL)

# This triggers partial computation for null checks:
lf = scan_parquet("users.parquet", Users)
```

### Full validation (`FULL`)

Full validation adds value-level constraint checks (`Field()` constraints, `@schema_check`). With Dask, this **materializes the entire DataFrame** because constraints like `ge=0`, `pattern=...`, and `unique=True` must inspect actual data values.

```python
cn.set_validation(cn.ValidationLevel.FULL)

# This triggers full computation:
lf = scan_parquet("users.parquet", Users)
```

!!! warning "Performance impact"
    `FULL` validation on a Dask DataFrame defeats the purpose of lazy evaluation — it forces the entire dataset into memory. Use `FULL` validation only on small datasets or after `.collect()`.

### Explicit `validate()`

Calling `lf.validate()` always runs both structural and value-level checks, regardless of the validation level toggle. On a Dask DataFrame, **this materializes the entire dataset**.

```python
# Careful — this computes the full dataset:
lf.validate()
```

### Recommendation

For Dask workflows, the recommended approach is:

1. Use `ValidationLevel.OFF` (the default) during pipeline construction
2. Validate a sample or after collection:

```python
# Option 1: Validate after collecting
result = lf.filter(Users.age > 25).collect()
result.validate()

# Option 2: Validate a sample
sample = lf.head(1000).collect().validate()
```

## `Field(unique=True)` on partitioned data

The `unique` constraint checks for duplicate values in a column. With Dask, validation materializes the full dataset before checking, so uniqueness is checked **globally** (not per-partition).

However, this means the entire dataset must fit in memory on a single worker during validation. For very large datasets where global uniqueness is important, consider checking uniqueness in your pipeline logic rather than via `Field(unique=True)`.

## `cast_schema` is fully deferred

`cast_schema()` applies column renames and selection lazily — no computation is triggered. The column mapping is resolved and applied as Dask task graph operations.

```python
class UserSummary(cn.Schema):
    user_name: cn.Column[cn.Utf8] = cn.mapped_from(Users.name)
    user_id: cn.Column[cn.UInt64] = cn.mapped_from(Users.id)

# No computation — just builds rename tasks:
summary = lf.cast_schema(UserSummary)

# Computation happens here:
result = summary.collect()
```

## I/O API

Since Dask is always lazy, only `scan_*` functions are provided:

| Function | Returns | Notes |
|----------|---------|-------|
| `scan_parquet` | `LazyFrame[S]` | Lazy scan with optional validation |
| `scan_csv` | `LazyFrame[S]` | Lazy scan with optional validation |
| `write_parquet` | `None` | Writes from DataFrame or LazyFrame |
| `write_csv` | `None` | Writes from DataFrame or LazyFrame |
