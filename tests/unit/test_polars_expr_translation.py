"""Unit tests for PolarsBackend expression translation."""

from __future__ import annotations

import polars as pl
import pytest

from colnade import Column, Float64, Schema, UInt64, Utf8, lit
from colnade.expr import (
    ColumnRef,
    FunctionCall,
    ListOp,
    Literal,
    StructFieldAccess,
    UnaryOp,
)
from colnade_polars.adapter import PolarsBackend

# ---------------------------------------------------------------------------
# Fixture schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class Address(Schema):
    street: Column[Utf8]
    city: Column[Utf8]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

backend = PolarsBackend()


def _eval(pl_expr: pl.Expr, df: pl.DataFrame | None = None) -> pl.Expr:
    """Return the Polars expression (for structural checks)."""
    return pl_expr


# ---------------------------------------------------------------------------
# ColumnRef
# ---------------------------------------------------------------------------


class TestColumnRef:
    def test_column_ref(self) -> None:
        expr = ColumnRef(column=Users.id)
        result = backend.translate_expr(expr)
        # Apply to a real df to verify it works
        df = pl.DataFrame({"id": [1, 2, 3]})
        series = df.select(result).to_series()
        assert series.to_list() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Literal
# ---------------------------------------------------------------------------


class TestLiteral:
    def test_int_literal(self) -> None:
        expr = Literal(value=42)
        result = backend.translate_expr(expr)
        df = pl.DataFrame({"x": [1]})
        val = df.select(result).item()
        assert val == 42

    def test_str_literal(self) -> None:
        expr = Literal(value="hello")
        result = backend.translate_expr(expr)
        df = pl.DataFrame({"x": [1]})
        val = df.select(result).item()
        assert val == "hello"

    def test_float_literal(self) -> None:
        expr = Literal(value=3.14)
        result = backend.translate_expr(expr)
        df = pl.DataFrame({"x": [1]})
        val = df.select(result).item()
        assert val == pytest.approx(3.14)

    def test_bool_literal(self) -> None:
        expr = Literal(value=True)
        result = backend.translate_expr(expr)
        df = pl.DataFrame({"x": [1]})
        val = df.select(result).item()
        assert val is True


# ---------------------------------------------------------------------------
# BinOp
# ---------------------------------------------------------------------------


class TestBinOp:
    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"id": [10, 20, 30], "age": [1, 2, 3]})

    def test_add(self, df: pl.DataFrame) -> None:
        expr = Users.id + 5
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [15, 25, 35]

    def test_sub(self, df: pl.DataFrame) -> None:
        expr = Users.id - 5
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [5, 15, 25]

    def test_mul(self, df: pl.DataFrame) -> None:
        expr = Users.id * 2
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [20, 40, 60]

    def test_truediv(self, df: pl.DataFrame) -> None:
        expr = Users.id / 10
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [1.0, 2.0, 3.0]

    def test_mod(self, df: pl.DataFrame) -> None:
        expr = Users.id % 15
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [10, 5, 0]

    def test_gt(self, df: pl.DataFrame) -> None:
        expr = Users.id > 15
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, True]

    def test_lt(self, df: pl.DataFrame) -> None:
        expr = Users.id < 25
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, True, False]

    def test_ge(self, df: pl.DataFrame) -> None:
        expr = Users.id >= 20
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, True]

    def test_le(self, df: pl.DataFrame) -> None:
        expr = Users.id <= 20
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, True, False]

    def test_eq(self, df: pl.DataFrame) -> None:
        expr = Users.id == 20
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, False]

    def test_ne(self, df: pl.DataFrame) -> None:
        expr = Users.id != 20
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, False, True]

    def test_and(self, df: pl.DataFrame) -> None:
        expr = (Users.id > 10) & (Users.id < 30)
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, False]

    def test_or(self, df: pl.DataFrame) -> None:
        expr = (Users.id == 10) | (Users.id == 30)
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, False, True]

    def test_column_to_column(self, df: pl.DataFrame) -> None:
        expr = Users.id + Users.age
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [11, 22, 33]


# ---------------------------------------------------------------------------
# UnaryOp
# ---------------------------------------------------------------------------


class TestUnaryOp:
    def test_neg(self) -> None:
        expr = -Users.age
        df = pl.DataFrame({"age": [1, 2, 3]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [-1, -2, -3]

    def test_not(self) -> None:
        inner = Users.id > 1
        expr = ~inner
        df = pl.DataFrame({"id": [1, 2, 3]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, False, False]

    def test_is_null(self) -> None:
        expr = Users.name.is_null()
        df = pl.DataFrame({"name": ["a", None, "c"]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, False]

    def test_is_not_null(self) -> None:
        expr = Users.name.is_not_null()
        df = pl.DataFrame({"name": ["a", None, "c"]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, False, True]

    def test_is_nan(self) -> None:
        expr = UnaryOp(operand=ColumnRef(column=Users.age), op="is_nan")
        df = pl.DataFrame({"age": [1.0, float("nan"), 3.0]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, False]


# ---------------------------------------------------------------------------
# Agg
# ---------------------------------------------------------------------------


class TestAgg:
    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"id": [1, 2, 3, 4, 5], "age": [10, 20, 30, 40, 50]})

    def test_sum(self, df: pl.DataFrame) -> None:
        expr = Users.age.sum()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == 150

    def test_mean(self, df: pl.DataFrame) -> None:
        expr = Users.age.mean()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == pytest.approx(30.0)

    def test_min(self, df: pl.DataFrame) -> None:
        expr = Users.age.min()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == 10

    def test_max(self, df: pl.DataFrame) -> None:
        expr = Users.age.max()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == 50

    def test_count(self, df: pl.DataFrame) -> None:
        expr = Users.age.count()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == 5

    def test_std(self, df: pl.DataFrame) -> None:
        expr = Users.age.std()
        result = df.select(backend.translate_expr(expr)).item()
        assert isinstance(result, float)

    def test_var(self, df: pl.DataFrame) -> None:
        expr = Users.age.var()
        result = df.select(backend.translate_expr(expr)).item()
        assert isinstance(result, float)

    def test_first(self, df: pl.DataFrame) -> None:
        expr = Users.age.first()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == 10

    def test_last(self, df: pl.DataFrame) -> None:
        expr = Users.age.last()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == 50

    def test_n_unique(self, df: pl.DataFrame) -> None:
        expr = Users.age.n_unique()
        result = df.select(backend.translate_expr(expr)).item()
        assert result == 5


# ---------------------------------------------------------------------------
# AliasedExpr
# ---------------------------------------------------------------------------


class TestAliasedExpr:
    def test_alias(self) -> None:
        expr = (Users.age * 2).alias(Users.id)
        df = pl.DataFrame({"age": [10, 20]})
        result = df.select(backend.translate_expr(expr))
        assert result.columns == ["id"]
        assert result.to_series().to_list() == [20, 40]


# ---------------------------------------------------------------------------
# FunctionCall — string methods
# ---------------------------------------------------------------------------


class TestStringFunctions:
    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"name": ["Alice", "Bob", "Charlie"]})

    def test_str_contains(self, df: pl.DataFrame) -> None:
        expr = Users.name.str_contains("li")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, False, True]

    def test_str_starts_with(self, df: pl.DataFrame) -> None:
        expr = Users.name.str_starts_with("Al")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [True, False, False]

    def test_str_ends_with(self, df: pl.DataFrame) -> None:
        expr = Users.name.str_ends_with("ob")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, False]

    def test_str_len(self, df: pl.DataFrame) -> None:
        expr = Users.name.str_len()
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [5, 3, 7]

    def test_str_to_lowercase(self, df: pl.DataFrame) -> None:
        expr = Users.name.str_to_lowercase()
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == ["alice", "bob", "charlie"]

    def test_str_to_uppercase(self, df: pl.DataFrame) -> None:
        expr = Users.name.str_to_uppercase()
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == ["ALICE", "BOB", "CHARLIE"]

    def test_str_strip(self) -> None:
        expr = Users.name.str_strip()
        df = pl.DataFrame({"name": [" a ", " b", "c "]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == ["a", "b", "c"]

    def test_str_replace(self, df: pl.DataFrame) -> None:
        expr = Users.name.str_replace("li", "LI")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == ["ALIce", "Bob", "CharLIe"]


# ---------------------------------------------------------------------------
# FunctionCall — temporal methods
# ---------------------------------------------------------------------------


class TestTemporalFunctions:
    @pytest.fixture
    def df(self) -> pl.DataFrame:
        from datetime import datetime

        return pl.DataFrame(
            {"ts": [datetime(2024, 3, 15, 10, 30, 45), datetime(2024, 12, 1, 23, 59, 0)]}
        )

    def test_dt_year(self, df: pl.DataFrame) -> None:
        ts_col = Column("ts", None, Users)
        fc = FunctionCall(name="dt_year", args=(ColumnRef(column=ts_col),))
        result = df.select(backend.translate_expr(fc)).to_series()
        assert result.to_list() == [2024, 2024]

    def test_dt_month(self, df: pl.DataFrame) -> None:
        result = df.select(
            backend.translate_expr(
                FunctionCall(name="dt_month", args=(ColumnRef(column=Column("ts", None, Users)),))
            )
        ).to_series()
        assert result.to_list() == [3, 12]

    def test_dt_day(self, df: pl.DataFrame) -> None:
        result = df.select(
            backend.translate_expr(
                FunctionCall(name="dt_day", args=(ColumnRef(column=Column("ts", None, Users)),))
            )
        ).to_series()
        assert result.to_list() == [15, 1]

    def test_dt_hour(self, df: pl.DataFrame) -> None:
        result = df.select(
            backend.translate_expr(
                FunctionCall(name="dt_hour", args=(ColumnRef(column=Column("ts", None, Users)),))
            )
        ).to_series()
        assert result.to_list() == [10, 23]

    def test_dt_minute(self, df: pl.DataFrame) -> None:
        result = df.select(
            backend.translate_expr(
                FunctionCall(name="dt_minute", args=(ColumnRef(column=Column("ts", None, Users)),))
            )
        ).to_series()
        assert result.to_list() == [30, 59]

    def test_dt_second(self, df: pl.DataFrame) -> None:
        result = df.select(
            backend.translate_expr(
                FunctionCall(name="dt_second", args=(ColumnRef(column=Column("ts", None, Users)),))
            )
        ).to_series()
        assert result.to_list() == [45, 0]


# ---------------------------------------------------------------------------
# FunctionCall — null/nan/cast/over
# ---------------------------------------------------------------------------


class TestMiscFunctions:
    def test_fill_null(self) -> None:
        expr = Users.name.fill_null("unknown")
        df = pl.DataFrame({"name": ["Alice", None, "Charlie"]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == ["Alice", "unknown", "Charlie"]

    def test_fill_nan(self) -> None:
        expr = FunctionCall(
            name="fill_nan",
            args=(ColumnRef(column=Column("val", None, Users)), Literal(value=0.0)),
        )
        df = pl.DataFrame({"val": [1.0, float("nan"), 3.0]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [1.0, 0.0, 3.0]

    def test_assert_non_null(self) -> None:
        expr = Users.name.assert_non_null()
        df = pl.DataFrame({"name": ["Alice", "Bob"]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == ["Alice", "Bob"]

    def test_cast(self) -> None:

        expr = Users.age.cast(Float64)
        df = pl.DataFrame({"age": [1, 2, 3]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.dtype == pl.Float64
        assert result.to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# StructFieldAccess
# ---------------------------------------------------------------------------


class TestStructFieldAccess:
    def test_struct_field(self) -> None:
        expr = StructFieldAccess(
            struct_expr=ColumnRef(column=Column("addr", None, Users)),
            field=Address.city,
        )
        df = pl.DataFrame(
            {"addr": [{"street": "123 Main", "city": "NYC"}, {"street": "456 Elm", "city": "LA"}]}
        )
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == ["NYC", "LA"]


# ---------------------------------------------------------------------------
# ListOp
# ---------------------------------------------------------------------------


class TestListOp:
    _tags_ref = ColumnRef(column=Column("tags", None, Users))

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        return pl.DataFrame({"tags": [[1, 2, 3], [4, 5], [6]]})

    def test_list_len(self, df: pl.DataFrame) -> None:
        expr = ListOp(list_expr=self._tags_ref, op="len")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [3, 2, 1]

    def test_list_get(self, df: pl.DataFrame) -> None:
        expr = ListOp(list_expr=self._tags_ref, op="get", args=(0,))
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [1, 4, 6]

    def test_list_contains(self, df: pl.DataFrame) -> None:
        expr = ListOp(list_expr=self._tags_ref, op="contains", args=(5,))
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, False]

    def test_list_sum(self, df: pl.DataFrame) -> None:
        expr = ListOp(list_expr=self._tags_ref, op="sum")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [6, 9, 6]

    def test_list_mean(self, df: pl.DataFrame) -> None:
        expr = ListOp(list_expr=self._tags_ref, op="mean")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [
            pytest.approx(2.0),
            pytest.approx(4.5),
            pytest.approx(6.0),
        ]

    def test_list_min(self, df: pl.DataFrame) -> None:
        expr = ListOp(list_expr=self._tags_ref, op="min")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [1, 4, 6]

    def test_list_max(self, df: pl.DataFrame) -> None:
        expr = ListOp(list_expr=self._tags_ref, op="max")
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [3, 5, 6]


# ---------------------------------------------------------------------------
# Nested expressions
# ---------------------------------------------------------------------------


class TestNestedExpressions:
    def test_complex_arithmetic(self) -> None:
        expr = (Users.age + 10) * 2 - lit(5)
        df = pl.DataFrame({"age": [10, 20]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [35, 55]

    def test_chained_comparison(self) -> None:
        expr = (Users.id > 1) & (Users.id < 5) & (Users.id != 3)
        df = pl.DataFrame({"id": [1, 2, 3, 4, 5]})
        result = df.select(backend.translate_expr(expr)).to_series()
        assert result.to_list() == [False, True, False, True, False]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unsupported_unary_op(self) -> None:
        expr = UnaryOp(operand=ColumnRef(column=Users.id), op="unknown")
        with pytest.raises(ValueError, match="Unsupported UnaryOp"):
            backend.translate_expr(expr)

    def test_unsupported_function_call(self) -> None:
        expr = FunctionCall(name="unknown_fn", args=())
        with pytest.raises(ValueError, match="Unsupported FunctionCall"):
            backend.translate_expr(expr)

    def test_unsupported_list_op(self) -> None:
        expr = ListOp(list_expr=ColumnRef(column=Users.id), op="unknown_op")
        with pytest.raises(ValueError, match="Unsupported ListOp"):
            backend.translate_expr(expr)
