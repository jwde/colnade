# Dask Backend

The Dask backend (`colnade-dask`) lets you use Colnade's typed DataFrame API on Dask's distributed DataFrames. This page covers Dask-specific semantics that differ from the Polars and Pandas backends.

## Dask is inherently lazy

Unlike Polars (which has separate eager `DataFrame` and lazy `LazyFrame`), Dask DataFrames are always lazy — operations build a task graph that executes only when you call `.compute()`.

This has implications for Colnade's API:

- **`DataFrame[S]`** wraps a Dask DataFrame, but operations are still deferred. Methods like `filter()`, `sort()`, `select()` return new Dask DataFrames with updated task graphs.
- **`LazyFrame[S]`** is functionally identical to `DataFrame[S]` under the Dask backend. The `lazy()` method is a no-op passthrough. Both are lazy.
- **`.collect()`** is where computation happens. It calls Dask's `.compute()`, materializing the result.

```python
from colnade_dask import read_parquet

df = read_parquet("users.parquet", Users)  # Dask DataFrame (lazy)
filtered = df.filter(Users.age > 25)       # Still lazy — builds task graph
result = filtered.to_native().compute()    # Triggers computation
```

## Which operations trigger computation

Most operations are deferred. A few must materialize data:

| Operation | Triggers Compute | Notes |
|-----------|:---:|-------|
| `filter`, `sort`, `limit` | No | Builds task graph |
| `unique`, `drop_nulls` | No | Builds task graph |
| `with_columns`, `select` | No | Builds task graph |
| `group_by().agg()` | No | Builds task graph |
| `join` | No | Builds task graph |
| `cast_schema` | No | Lazy rename + column selection |
| `head(n)` | No | Uses Dask's `head(compute=False)` |
| `tail(n)` | **Yes** | Materializes to get tail rows |
| `sample(n)` | **Yes** | Dask doesn't support fixed-count sampling on partitions |
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
import colnade
from colnade import ValidationLevel

colnade.set_validation(ValidationLevel.STRUCTURAL)

# This triggers partial computation for null checks:
df = read_parquet("users.parquet", Users)
```

### Full validation (`FULL`)

Full validation adds value-level constraint checks (`Field()` constraints, `@schema_check`). With Dask, this **materializes the entire DataFrame** because constraints like `ge=0`, `pattern=...`, and `unique=True` must inspect actual data values.

```python
colnade.set_validation(ValidationLevel.FULL)

# This triggers full computation:
df = read_parquet("users.parquet", Users)
```

!!! warning "Performance impact"
    `FULL` validation on a Dask DataFrame defeats the purpose of lazy evaluation — it forces the entire dataset into memory. Use `FULL` validation only on small datasets or after `.collect()`.

### Explicit `validate()`

Calling `df.validate()` always runs both structural and value-level checks, regardless of the validation level toggle. On a Dask DataFrame, **this materializes the entire dataset**.

```python
# Careful — this computes the full dataset:
df.validate()
```

### Recommendation

For Dask workflows, the recommended approach is:

1. Use `ValidationLevel.OFF` (the default) during pipeline construction
2. Validate a sample or after collection:

```python
# Option 1: Validate after collecting
result = df.filter(Users.age > 25).collect()
result.validate()

# Option 2: Validate a sample
sample_df = df.head(1000).validate()
```

## `Field(unique=True)` on partitioned data

The `unique` constraint checks for duplicate values in a column. With Dask, validation materializes the full dataset before checking, so uniqueness is checked **globally** (not per-partition).

However, this means the entire dataset must fit in memory on a single worker during validation. For very large datasets where global uniqueness is important, consider checking uniqueness in your pipeline logic rather than via `Field(unique=True)`.

## `cast_schema` is fully deferred

`cast_schema()` applies column renames and selection lazily — no computation is triggered. The column mapping is resolved and applied as Dask task graph operations.

```python
class UserSummary(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Users.id)

# No computation — just builds rename tasks:
summary = df.cast_schema(UserSummary)

# Computation happens here:
result = summary.to_native().compute()
```

## `read_*` vs `scan_*`

Both `read_parquet`/`read_csv` and `scan_parquet`/`scan_csv` return Dask DataFrames, which are inherently lazy. The distinction exists for API consistency with the Polars backend (where `read_*` is eager and `scan_*` is lazy), but with Dask both behave identically — data is only read when computation is triggered.

| Function | Returns | Behavior |
|----------|---------|----------|
| `read_parquet` | `DataFrame[S]` | Lazy (deferred read) |
| `scan_parquet` | `LazyFrame[S]` | Lazy (deferred read) |
| `read_csv` | `DataFrame[S]` | Lazy (deferred read) |
| `scan_csv` | `LazyFrame[S]` | Lazy (deferred read) |
