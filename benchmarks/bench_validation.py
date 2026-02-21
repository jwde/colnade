"""Benchmark: Validation overhead at each level (OFF / STRUCTURAL / FULL).

Measures the cost of enabling validation at different levels across all
three backends.

    uv run python benchmarks/bench_validation.py
"""

from __future__ import annotations

import timeit
from dataclasses import dataclass

import dask.dataframe as dd
import pandas as pd
import polars as pl

import colnade
from colnade import Column, Field, Float64, Schema, UInt64, Utf8, ValidationLevel
from colnade.dataframe import DataFrame
from colnade_dask import DaskBackend
from colnade_pandas import PandasBackend
from colnade_polars import PolarsBackend

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    """Schema without Field() constraints — used for OFF and STRUCTURAL."""

    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class UsersConstrained(Schema):
    """Schema with Field() constraints — used for FULL validation."""

    id: Column[UInt64] = Field(unique=True)
    name: Column[Utf8] = Field(min_length=1, max_length=100)
    age: Column[UInt64] = Field(ge=0, le=150)
    score: Column[Float64] = Field(ge=0.0, le=100.0)


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


def make_polars(n: int) -> pl.DataFrame:
    return pl.DataFrame(_base_data(n)).cast({"id": pl.UInt64, "age": pl.UInt64})


def make_pandas(n: int) -> pd.DataFrame:
    raw = pd.DataFrame(_base_data(n))
    raw["id"] = raw["id"].astype(pd.UInt64Dtype())
    raw["name"] = raw["name"].astype(pd.StringDtype())
    raw["age"] = raw["age"].astype(pd.UInt64Dtype())
    raw["score"] = raw["score"].astype(pd.Float64Dtype())
    return raw


def make_dask(n: int) -> dd.DataFrame:
    pdf = make_pandas(n)
    return dd.from_pandas(pdf, npartitions=max(1, n // 10_000))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VResult:
    label: str
    off_us: float
    structural_us: float
    full_us: float

    @property
    def structural_overhead(self) -> str:
        if self.off_us == 0:
            return "N/A"
        pct = ((self.structural_us - self.off_us) / self.off_us) * 100
        return f"+{pct:.0f}%"

    @property
    def full_overhead(self) -> str:
        if self.off_us == 0:
            return "N/A"
        pct = ((self.full_us - self.off_us) / self.off_us) * 100
        return f"+{pct:.0f}%"


# ---------------------------------------------------------------------------
# Benchmark helpers — measure validate_schema / validate_field_constraints
# ---------------------------------------------------------------------------


def _bench_validation(backend, data, n: int, iters: int, label_prefix: str) -> VResult:
    """Measure validation cost at each level for a given backend and data."""

    # OFF — just construct the DataFrame, no validation
    def off_op():
        return DataFrame(_data=data, _schema=Users, _backend=backend)

    t_off = timeit.timeit(off_op, number=iters) / iters * 1e6

    # STRUCTURAL — construct + validate_schema
    def structural_op():
        backend.validate_schema(data, Users)

    t_structural = timeit.timeit(structural_op, number=iters) / iters * 1e6

    # FULL — validate_schema + validate_field_constraints
    def full_op():
        backend.validate_schema(data, UsersConstrained)
        backend.validate_field_constraints(data, UsersConstrained)

    t_full = timeit.timeit(full_op, number=iters) / iters * 1e6

    return VResult(f"{label_prefix}, {n:,} rows", t_off, t_structural, t_full)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    sizes = [100, 10_000, 1_000_000]

    print()
    print("=== Validation Overhead by Level ===")
    print()
    cols = [
        f"{'Benchmark':<45}",
        f"{'OFF (us)':>10}",
        f"{'STRUCT (us)':>12}",
        f"{'FULL (us)':>10}",
        f"{'STRUCT/OFF':>11}",
        f"{'FULL/OFF':>9}",
    ]
    hdr = " ".join(cols)
    print(hdr)
    print("-" * len(hdr))

    # Make sure validation is off globally (we call backend methods directly)
    colnade.set_validation(ValidationLevel.OFF)

    for n in sizes:
        iters = max(10, 200 // (n // 100 + 1))

        polars_data = make_polars(n)
        pandas_data = make_pandas(n)
        dask_data = make_dask(n)

        for backend, data, name in [
            (PolarsBackend(), polars_data, "Polars validate"),
            (PandasBackend(), pandas_data, "Pandas validate"),
            (DaskBackend(), dask_data, "Dask validate"),
        ]:
            r = _bench_validation(backend, data, n, iters, name)
            print(
                f"{r.label:<45} {r.off_us:>10.0f} {r.structural_us:>12.0f} "
                f"{r.full_us:>10.0f} {r.structural_overhead:>11} {r.full_overhead:>9}"
            )

        print()


if __name__ == "__main__":
    main()
