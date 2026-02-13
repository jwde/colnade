# Benchmarks

Measures the runtime overhead of Colnade's expression AST construction and translation compared to calling Polars directly.

## Running

```bash
uv run python benchmarks/bench_overhead.py
```

## What's measured

| Benchmark | Description |
|-----------|-------------|
| Expr construction | `Users.age > 25` vs `pl.col("age") > 25` |
| filter | Typed `df.filter(Users.age > 40)` vs `df.filter(pl.col("age") > 40)` |
| select | Typed `df.select(Users.name, Users.score)` vs `df.select("name", "score")` |
| pipeline | filter + sort + select chain |
| lazy pipeline | Same chain via LazyFrame with `.collect()` |

Each benchmark is run at 100, 10K, and 1M rows.

## Typical results

Overhead is negligible (0-5%) for most operations. The AST construction cost is constant and dwarfed by Polars' own execution time, even at small DataFrame sizes.
