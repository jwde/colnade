# Performance

Colnade's expression DSL builds an AST (abstract syntax tree) that gets translated to engine-native operations. This page shows the overhead of that abstraction layer.

## Key takeaway

**Colnade adds < 5% overhead for typical operations.** The AST construction and translation cost is constant per operation and negligible compared to engine execution time, even at small DataFrame sizes.

## Benchmark results

All benchmarks compare Colnade-typed operations against equivalent raw Polars calls. Times are in microseconds (us). Run on a single core, Python 3.10, Polars 1.x.

### Expression construction

| Operation | Raw Polars | Colnade | Overhead |
|-----------|-----------|---------|----------|
| `col > 25` | ~3 us | ~3 us | < 1 us |

Expression construction (building the AST vs building a Polars expression) has negligible cost — both are object allocations with no I/O or computation.

### Single operations

| Operation | Rows | Raw Polars (us) | Colnade (us) | Overhead |
|-----------|------|-----------------|--------------|----------|
| filter | 100 | 250 | 260 | +2-4% |
| filter | 10K | 245 | 250 | +2% |
| filter | 1M | 2,100 | 2,300 | +5% |
| select | 100 | 105 | 110 | +5% |
| select | 10K | 110 | 110 | ~0% |
| select | 1M | 115 | 115 | ~0% |

### Pipeline (filter + sort + select)

| Rows | Raw Polars (us) | Colnade (us) | Overhead |
|------|-----------------|--------------|----------|
| 100 | 600 | 650 | +5-8% |
| 10K | 2,660 | 2,820 | +5% |
| 1M | 20,000 | 20,000 | ~0% |

### Lazy pipeline (build + collect)

| Rows | Raw Polars (us) | Colnade (us) | Overhead |
|------|-----------------|--------------|----------|
| 100 | 320 | 340 | +6% |
| 10K | 1,560 | 1,580 | +1% |
| 1M | 17,500 | 18,000 | +2% |

## Why overhead is low

Colnade's overhead comes from two sources, both constant per operation:

1. **AST construction** — Building `BinOp(ColumnRef("age"), Literal(25), ">")` instead of `pl.col("age") > 25`. Both are pure Python object allocations — the difference is a few microseconds.

2. **AST translation** — Walking the expression tree and producing engine-native calls. This is a simple recursive function with no I/O.

Neither source grows with DataFrame size. As rows increase, the engine's execution time dominates and the abstraction cost disappears into measurement noise.

## Running benchmarks yourself

```bash
uv run python benchmarks/bench_overhead.py
```

The benchmark script runs each operation 100-1000 times and reports mean times. Results vary by hardware — the relative overhead percentages are more meaningful than absolute times.

## Validation overhead

The benchmarks above run with validation **disabled** (the default). When validation is enabled:

- **`STRUCTURAL`** — Adds schema checking at I/O boundaries. Cost depends on number of columns, not rows. Typically < 100 us per validation call.
- **`FULL`** — Adds value-level constraint checking. Cost depends on the number of constraints and rows, as each constraint scans the relevant column.

Validation is designed for development and CI, not production hot paths. Use `ValidationLevel.OFF` (the default) in production.
