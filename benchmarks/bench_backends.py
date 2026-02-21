"""Benchmark: Colnade expression overhead across all backends.

Measures the cost of Colnade's abstraction layer compared to calling each
backend directly. Covers Polars, Pandas, and Dask.

    uv run python benchmarks/bench_backends.py
"""

from __future__ import annotations

import timeit
from dataclasses import dataclass

import dask.dataframe as dd
import pandas as pd
import polars as pl

from colnade import Column, Float64, Schema, UInt64, Utf8
from colnade.dataframe import DataFrame
from colnade_dask import DaskBackend
from colnade_pandas import PandasBackend
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
# Data helpers
# ---------------------------------------------------------------------------


def _base_data(n: int) -> dict:
    return {
        "id": list(range(n)),
        "name": [f"user_{i}" for i in range(n)],
        "age": [20 + (i % 60) for i in range(n)],
        "score": [50.0 + (i % 50) for i in range(n)],
    }


def make_polars(n: int) -> tuple[pl.DataFrame, DataFrame]:
    data = _base_data(n)
    raw = pl.DataFrame(data).cast({"id": pl.UInt64, "age": pl.UInt64})
    typed = DataFrame(_data=raw, _schema=Users, _backend=PolarsBackend())
    return raw, typed


def make_pandas(n: int) -> tuple[pd.DataFrame, DataFrame]:
    raw = pd.DataFrame(_base_data(n))
    raw["id"] = raw["id"].astype(pd.UInt64Dtype())
    raw["name"] = raw["name"].astype(pd.StringDtype())
    raw["age"] = raw["age"].astype(pd.UInt64Dtype())
    raw["score"] = raw["score"].astype(pd.Float64Dtype())
    typed = DataFrame(_data=raw.copy(), _schema=Users, _backend=PandasBackend())
    return raw, typed


def make_dask(n: int) -> tuple[dd.DataFrame, DataFrame]:
    pdf = pd.DataFrame(_base_data(n))
    pdf["id"] = pdf["id"].astype(pd.UInt64Dtype())
    pdf["name"] = pdf["name"].astype(pd.StringDtype())
    pdf["age"] = pdf["age"].astype(pd.UInt64Dtype())
    pdf["score"] = pdf["score"].astype(pd.Float64Dtype())
    nparts = max(1, n // 10_000)
    raw = dd.from_pandas(pdf, npartitions=nparts)
    typed = DataFrame(
        _data=dd.from_pandas(pdf, npartitions=nparts),
        _schema=Users,
        _backend=DaskBackend(),
    )
    return raw, typed


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    label: str
    raw_us: float
    colnade_us: float

    @property
    def overhead_pct(self) -> float:
        return ((self.colnade_us - self.raw_us) / self.raw_us) * 100 if self.raw_us > 0 else 0.0


# ---------------------------------------------------------------------------
# Polars benchmarks
# ---------------------------------------------------------------------------


def bench_polars(n: int, iters: int) -> list[BenchResult]:
    raw, typed = make_polars(n)
    results = []

    # filter
    t_raw = timeit.timeit(lambda: raw.filter(pl.col("age") > 40), number=iters)
    t_col = timeit.timeit(lambda: typed.filter(Users.age > 40), number=iters)
    results.append(
        BenchResult(f"Polars filter, {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    # select
    t_raw = timeit.timeit(lambda: raw.select("name", "score"), number=iters)
    t_col = timeit.timeit(lambda: typed.select(Users.name, Users.score), number=iters)
    results.append(
        BenchResult(f"Polars select, {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    # pipeline: filter + sort + select
    def raw_pipe():
        return raw.filter(pl.col("age") > 30).sort("score", descending=True).select("name", "score")

    def col_pipe():
        return typed.filter(Users.age > 30).sort(Users.score.desc()).select(Users.name, Users.score)

    t_raw = timeit.timeit(raw_pipe, number=iters)
    t_col = timeit.timeit(col_pipe, number=iters)
    results.append(
        BenchResult(f"Polars pipeline, {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    return results


# ---------------------------------------------------------------------------
# Pandas benchmarks
# ---------------------------------------------------------------------------


def bench_pandas(n: int, iters: int) -> list[BenchResult]:
    raw, typed = make_pandas(n)
    results = []

    # filter — raw Pandas uses boolean indexing directly
    t_raw = timeit.timeit(lambda: raw.loc[raw["age"] > 40].reset_index(drop=True), number=iters)
    t_col = timeit.timeit(lambda: typed.filter(Users.age > 40), number=iters)
    results.append(
        BenchResult(f"Pandas filter, {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    # select
    t_raw = timeit.timeit(lambda: raw[["name", "score"]].reset_index(drop=True), number=iters)
    t_col = timeit.timeit(lambda: typed.select(Users.name, Users.score), number=iters)
    results.append(
        BenchResult(f"Pandas select, {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    # pipeline: filter + sort + select
    def raw_pipe():
        return (
            raw.loc[raw["age"] > 30]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)[["name", "score"]]
        )

    def col_pipe():
        return typed.filter(Users.age > 30).sort(Users.score.desc()).select(Users.name, Users.score)

    t_raw = timeit.timeit(raw_pipe, number=iters)
    t_col = timeit.timeit(col_pipe, number=iters)
    results.append(
        BenchResult(f"Pandas pipeline, {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    return results


# ---------------------------------------------------------------------------
# Dask benchmarks (lazy graph construction only — no .compute())
# ---------------------------------------------------------------------------


def bench_dask(n: int, iters: int) -> list[BenchResult]:
    raw, typed = make_dask(n)
    results = []

    # filter — both sides build lazy task graph
    t_raw = timeit.timeit(lambda: raw[raw["age"] > 40], number=iters)
    t_col = timeit.timeit(lambda: typed.filter(Users.age > 40), number=iters)
    results.append(
        BenchResult(f"Dask filter (lazy), {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    # select
    t_raw = timeit.timeit(lambda: raw[["name", "score"]], number=iters)
    t_col = timeit.timeit(lambda: typed.select(Users.name, Users.score), number=iters)
    results.append(
        BenchResult(f"Dask select (lazy), {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    # pipeline
    def raw_pipe():
        return raw[raw["age"] > 30].sort_values("score", ascending=False)[["name", "score"]]

    def col_pipe():
        return typed.filter(Users.age > 30).sort(Users.score.desc()).select(Users.name, Users.score)

    t_raw = timeit.timeit(raw_pipe, number=iters)
    t_col = timeit.timeit(col_pipe, number=iters)
    results.append(
        BenchResult(f"Dask pipeline (lazy), {n:,} rows", t_raw / iters * 1e6, t_col / iters * 1e6)
    )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    sizes = [100, 10_000, 1_000_000]

    print()
    print(f"{'Benchmark':<45} {'Raw (us)':>10} {'Colnade (us)':>13} {'Overhead':>10}")
    print("-" * 82)

    for n in sizes:
        iters = max(20, 500 // (n // 100 + 1))

        for results in [bench_polars(n, iters), bench_pandas(n, iters), bench_dask(n, iters)]:
            for r in results:
                overhead = f"+{r.overhead_pct:.0f}%"
                print(f"{r.label:<45} {r.raw_us:>10.0f} {r.colnade_us:>13.0f} {overhead:>10}")

        print()


if __name__ == "__main__":
    main()
