# Benchmarks

Measures the runtime overhead of Colnade's expression AST construction and translation compared to calling each backend directly.

## Running

```bash
# Polars-specific overhead (expression construction, filter, select, pipeline, lazy)
uv run python benchmarks/bench_overhead.py

# All three backends (Polars, Pandas, Dask)
uv run python benchmarks/bench_backends.py

# Validation overhead at each level (OFF / STRUCTURAL / FULL)
uv run python benchmarks/bench_validation.py
```

## What's measured

### `bench_overhead.py` — Polars expression overhead

| Benchmark | Description |
|-----------|-------------|
| Expr construction | `Users.age > 25` vs `pl.col("age") > 25` |
| filter | Typed `df.filter(Users.age > 40)` vs `df.filter(pl.col("age") > 40)` |
| select | Typed `df.select(Users.name, Users.score)` vs `df.select("name", "score")` |
| pipeline | filter + sort + select chain |
| lazy pipeline | Same chain via LazyFrame with `.collect()` |

### `bench_backends.py` — Cross-backend overhead

Runs filter, select, and pipeline benchmarks against all three backends (Polars, Pandas, Dask). Compares Colnade-typed operations against equivalent raw backend calls. Dask benchmarks measure lazy graph construction only (no `.compute()`).

### `bench_validation.py` — Validation cost by level

Measures the cost of `validate_schema()` (STRUCTURAL) and `validate_field_constraints()` (FULL) across all three backends. Uses a schema with `Field(unique=True, ge=..., le=..., min_length=..., max_length=...)` constraints for FULL benchmarks.

## Each benchmark is run at 100, 10K, and 1M rows.

## Typical results

- **Polars**: < 5% overhead. AST construction cost is constant and dwarfed by Polars' execution time.
- **Pandas**: < 5% for single operations; 10–25% for multi-step pipelines at large sizes due to callable indirection.
- **Dask**: ~200–300 us fixed overhead per operation on graph construction. Negligible vs actual `.compute()` time.
- **Validation**: Polars STRUCTURAL is ~20 us (constant). FULL scales with rows × constraints. Dask validation requires materialization.
