"""Benchmark: Colnade expression overhead vs raw Polars.

Measures the cost of Colnade's expression AST construction and translation
compared to calling Polars directly. Run with:

    uv run python benchmarks/bench_overhead.py

Results are printed as a table showing absolute times and relative overhead.
"""

from __future__ import annotations

import timeit
from dataclasses import dataclass

import polars as pl

from colnade import Column, Float64, Schema, UInt64, Utf8
from colnade_polars import PolarsBackend

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

backend = PolarsBackend()


def make_df(n: int) -> pl.DataFrame:
    """Create a raw Polars DataFrame with n rows."""
    return pl.DataFrame(
        {
            "id": range(n),
            "name": [f"user_{i}" for i in range(n)],
            "age": [20 + (i % 60) for i in range(n)],
            "score": [50.0 + (i % 50) for i in range(n)],
        }
    )


def make_typed_df(n: int):
    """Create a Colnade-typed DataFrame with n rows."""
    from colnade.dataframe import DataFrame

    raw = make_df(n)
    return DataFrame(_data=raw, _schema=Users, _backend=backend)


@dataclass
class BenchResult:
    label: str
    raw_us: float
    colnade_us: float

    @property
    def overhead_us(self) -> float:
        return self.colnade_us - self.raw_us

    @property
    def overhead_pct(self) -> float:
        return (self.overhead_us / self.raw_us) * 100 if self.raw_us > 0 else float("inf")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

ITERATIONS = 1000


def bench_expr_construction() -> BenchResult:
    """Measure expression construction: pl.col("age") > 25 vs Users.age > 25."""
    raw_time = timeit.timeit(lambda: pl.col("age") > 25, number=ITERATIONS)
    colnade_time = timeit.timeit(lambda: Users.age > 25, number=ITERATIONS)
    return BenchResult(
        "Expr construction (col > 25)",
        raw_time / ITERATIONS * 1_000_000,
        colnade_time / ITERATIONS * 1_000_000,
    )


def bench_filter(n: int) -> BenchResult:
    """Measure filter: raw Polars vs Colnade typed DataFrame."""
    raw_df = make_df(n)
    typed_df = make_typed_df(n)

    iters = max(100, ITERATIONS // (n // 100 + 1))

    raw_time = timeit.timeit(lambda: raw_df.filter(pl.col("age") > 40), number=iters)
    colnade_time = timeit.timeit(lambda: typed_df.filter(Users.age > 40), number=iters)

    return BenchResult(
        f"filter (age > 40), {n:,} rows",
        raw_time / iters * 1_000_000,
        colnade_time / iters * 1_000_000,
    )


def bench_select(n: int) -> BenchResult:
    """Measure select: raw Polars vs Colnade typed DataFrame."""
    raw_df = make_df(n)
    typed_df = make_typed_df(n)

    iters = max(100, ITERATIONS // (n // 100 + 1))

    raw_time = timeit.timeit(lambda: raw_df.select("name", "score"), number=iters)
    colnade_time = timeit.timeit(lambda: typed_df.select(Users.name, Users.score), number=iters)

    return BenchResult(
        f"select (2 cols), {n:,} rows",
        raw_time / iters * 1_000_000,
        colnade_time / iters * 1_000_000,
    )


def bench_pipeline(n: int) -> BenchResult:
    """Measure a filter+sort+select pipeline."""
    raw_df = make_df(n)
    typed_df = make_typed_df(n)

    iters = max(50, ITERATIONS // (n // 100 + 1))

    def raw_pipeline():
        return (
            raw_df.filter(pl.col("age") > 30).sort("score", descending=True).select("name", "score")
        )

    def colnade_pipeline():
        return (
            typed_df.filter(Users.age > 30).sort(Users.score.desc()).select(Users.name, Users.score)
        )

    raw_time = timeit.timeit(raw_pipeline, number=iters)
    colnade_time = timeit.timeit(colnade_pipeline, number=iters)

    return BenchResult(
        f"pipeline (filter+sort+select), {n:,} rows",
        raw_time / iters * 1_000_000,
        colnade_time / iters * 1_000_000,
    )


def bench_lazy_pipeline(n: int) -> BenchResult:
    """Measure lazy pipeline: build + collect."""
    raw_df = make_df(n)
    typed_df = make_typed_df(n)

    iters = max(50, ITERATIONS // (n // 100 + 1))

    def raw_lazy():
        return (
            raw_df.lazy()
            .filter(pl.col("age") > 30)
            .sort("score", descending=True)
            .select("name", "score")
            .collect()
        )

    def colnade_lazy():
        return (
            typed_df.lazy()
            .filter(Users.age > 30)
            .sort(Users.score.desc())
            .select(Users.name, Users.score)
            .collect()
        )

    raw_time = timeit.timeit(raw_lazy, number=iters)
    colnade_time = timeit.timeit(colnade_lazy, number=iters)

    return BenchResult(
        f"lazy pipeline (build+collect), {n:,} rows",
        raw_time / iters * 1_000_000,
        colnade_time / iters * 1_000_000,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    results: list[BenchResult] = []

    # Expression construction (no data)
    results.append(bench_expr_construction())

    # Varying DataFrame sizes
    for n in [100, 10_000, 1_000_000]:
        results.append(bench_filter(n))
        results.append(bench_select(n))
        results.append(bench_pipeline(n))
        results.append(bench_lazy_pipeline(n))

    # Print results
    print()
    print(f"{'Benchmark':<45} {'Raw (us)':>10} {'Colnade (us)':>13} {'Overhead':>10}")
    print("-" * 82)
    for r in results:
        if r.overhead_pct < 10000:
            overhead_str = f"+{r.overhead_pct:.0f}%"
        else:
            overhead_str = f"+{r.overhead_us:.0f}us"
        print(f"{r.label:<45} {r.raw_us:>10.1f} {r.colnade_us:>13.1f} {overhead_str:>10}")
    print()


if __name__ == "__main__":
    main()
