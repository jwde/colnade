"""Tests targeting uncovered code paths in the Polars backend adapter and I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

import colnade
from colnade import (
    Column,
    Schema,
    UInt64,
    Utf8,
    ValidationLevel,
)
from colnade_polars.adapter import PolarsBackend
from colnade_polars.io import read_parquet, scan_parquet

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class Events(Schema):
    id: Column[UInt64]
    ts: Column[colnade.Datetime]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_backend = PolarsBackend()


def _users_pldf() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "name": pl.Series(["Alice", "Bob", "Charlie"], dtype=pl.Utf8),
            "age": pl.Series([30, 25, 35], dtype=pl.UInt64),
        }
    )


# ---------------------------------------------------------------------------
# I/O validation level branches
# ---------------------------------------------------------------------------


class TestPolarsIOValidation:
    def test_read_parquet_with_structural_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.STRUCTURAL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.parquet")
                _users_pldf().write_parquet(path)
                result = read_parquet(path, Users)
                assert result._data.shape[0] == 3
        finally:
            colnade.set_validation(prev)

    def test_read_parquet_with_full_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.FULL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.parquet")
                _users_pldf().write_parquet(path)
                result = read_parquet(path, Users)
                assert result._data.shape[0] == 3
        finally:
            colnade.set_validation(prev)

    def test_scan_parquet_with_structural_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.STRUCTURAL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.parquet")
                _users_pldf().write_parquet(path)
                result = scan_parquet(path, Users)
                assert result._data.collect().shape[0] == 3
        finally:
            colnade.set_validation(prev)

    def test_scan_parquet_with_full_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.FULL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.parquet")
                _users_pldf().write_parquet(path)
                result = scan_parquet(path, Users)
                assert result._data.collect().shape[0] == 3
        finally:
            colnade.set_validation(prev)


# ---------------------------------------------------------------------------
# Adapter error branches
# ---------------------------------------------------------------------------


class TestAdapterErrorBranches:
    def test_unsupported_binop_raises(self) -> None:
        from colnade.expr import BinOp, ColumnRef, Literal

        bad_expr = BinOp(left=ColumnRef(column=Users.id), right=Literal(value=1), op="^^^")
        with pytest.raises(ValueError, match="Unsupported BinOp"):
            _backend.translate_expr(bad_expr)

    def test_unsupported_unaryop_raises(self) -> None:
        from colnade.expr import ColumnRef, UnaryOp

        bad_expr = UnaryOp(operand=ColumnRef(column=Users.id), op="bad_op")
        with pytest.raises(ValueError, match="Unsupported UnaryOp"):
            _backend.translate_expr(bad_expr)

    def test_unsupported_function_call_raises(self) -> None:
        from colnade.expr import ColumnRef, FunctionCall

        bad_expr = FunctionCall(name="bad_function", args=(ColumnRef(column=Users.id),))
        with pytest.raises(ValueError, match="Unsupported FunctionCall"):
            _backend.translate_expr(bad_expr)

    def test_unsupported_expression_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported expression type"):
            _backend.translate_expr("not an expr")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Function calls: dt_truncate, over
# ---------------------------------------------------------------------------


class TestPolarsSpecificPaths:
    def test_dt_truncate(self) -> None:
        from colnade_polars.io import from_dict

        df = from_dict(
            Events,
            {
                "id": [1, 2],
                "ts": [
                    "2025-03-15T10:30:45",
                    "2025-06-20T14:15:30",
                ],
            },
        )
        result = df.with_columns(Events.ts.dt_truncate("1d").alias(Events.ts))
        native = result.to_native()
        assert native["ts"][0].day == 15

    def test_over_window_with_agg(self) -> None:
        from colnade_polars.io import from_dict

        df = from_dict(
            Users,
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Alice", "Bob", "Bob"],
                "age": [30, 25, 35, 28],
            },
        )
        result = df.with_columns(Users.age.sum().over(Users.name).alias(Users.age))
        ages = result.to_native()["age"].to_list()
        # Alice group: 30+25=55, Bob group: 35+28=63
        assert ages == [55, 55, 63, 63]

    def test_over_window_no_agg(self) -> None:
        from colnade_polars.io import from_dict

        df = from_dict(
            Users,
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Alice", "Bob", "Bob"],
                "age": [30, 25, 35, 28],
            },
        )
        result = df.with_columns(Users.age.over(Users.name).alias(Users.age))
        ages = result.to_native()["age"].to_list()
        assert ages == [30, 25, 35, 28]
