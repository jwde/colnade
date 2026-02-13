"""PolarsBackend — translates Colnade expression trees and executes operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from colnade.expr import (
    Agg,
    AliasedExpr,
    BinOp,
    ColumnRef,
    Expr,
    FunctionCall,
    ListOp,
    Literal,
    SortExpr,
    StructFieldAccess,
    UnaryOp,
)
from colnade.schema import Column, Schema, SchemaError
from colnade_polars.conversion import map_colnade_dtype, map_polars_dtype

# ---------------------------------------------------------------------------
# BinOp operator dispatch
# ---------------------------------------------------------------------------

_BINOP_MAP: dict[str, str] = {
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
    "%": "__mod__",
    ">": "__gt__",
    "<": "__lt__",
    ">=": "__ge__",
    "<=": "__le__",
    "==": "__eq__",
    "!=": "__ne__",
    "&": "__and__",
    "|": "__or__",
}

# ---------------------------------------------------------------------------
# PolarsBackend
# ---------------------------------------------------------------------------


class PolarsBackend:
    """Colnade backend adapter for Polars."""

    # --- Expression translation ---

    def translate_expr(self, expr: Expr[Any]) -> pl.Expr:
        """Recursively translate a Colnade AST node to a Polars expression."""
        if isinstance(expr, AliasedExpr):
            inner = self.translate_expr(expr.expr)
            return inner.alias(expr.target.name)

        if isinstance(expr, ColumnRef):
            return pl.col(expr.column.name)

        if isinstance(expr, Literal):
            return pl.lit(expr.value)

        if isinstance(expr, BinOp):
            left = self.translate_expr(expr.left)
            right = self.translate_expr(expr.right)
            method = _BINOP_MAP.get(expr.op)
            if method is None:
                msg = f"Unsupported BinOp operator: {expr.op}"
                raise ValueError(msg)
            return getattr(left, method)(right)

        if isinstance(expr, UnaryOp):
            operand = self.translate_expr(expr.operand)
            if expr.op == "-":
                return -operand
            if expr.op == "~":
                return ~operand
            if expr.op == "is_null":
                return operand.is_null()
            if expr.op == "is_not_null":
                return operand.is_not_null()
            if expr.op == "is_nan":
                return operand.is_nan()
            msg = f"Unsupported UnaryOp: {expr.op}"
            raise ValueError(msg)

        if isinstance(expr, Agg):
            source = self.translate_expr(expr.source)
            return getattr(source, expr.agg_type)()

        if isinstance(expr, FunctionCall):
            return self._translate_function_call(expr)

        if isinstance(expr, StructFieldAccess):
            struct = self.translate_expr(expr.struct_expr)
            return struct.struct.field(expr.field.name)

        if isinstance(expr, ListOp):
            return self._translate_list_op(expr)

        msg = f"Unsupported expression type: {type(expr).__name__}"
        raise TypeError(msg)

    def _translate_function_call(self, expr: FunctionCall[Any]) -> pl.Expr:
        """Translate a FunctionCall node to Polars."""
        name = expr.name

        # String methods
        if name == "str_contains":
            source = self.translate_expr(expr.args[0])
            return source.str.contains(expr.args[1])
        if name == "str_starts_with":
            source = self.translate_expr(expr.args[0])
            return source.str.starts_with(expr.args[1])
        if name == "str_ends_with":
            source = self.translate_expr(expr.args[0])
            return source.str.ends_with(expr.args[1])
        if name == "str_len":
            source = self.translate_expr(expr.args[0])
            return source.str.len_chars()
        if name == "str_to_lowercase":
            source = self.translate_expr(expr.args[0])
            return source.str.to_lowercase()
        if name == "str_to_uppercase":
            source = self.translate_expr(expr.args[0])
            return source.str.to_uppercase()
        if name == "str_strip":
            source = self.translate_expr(expr.args[0])
            return source.str.strip_chars()
        if name == "str_replace":
            source = self.translate_expr(expr.args[0])
            return source.str.replace(expr.args[1], expr.args[2])

        # Temporal methods
        if name == "dt_year":
            source = self.translate_expr(expr.args[0])
            return source.dt.year()
        if name == "dt_month":
            source = self.translate_expr(expr.args[0])
            return source.dt.month()
        if name == "dt_day":
            source = self.translate_expr(expr.args[0])
            return source.dt.day()
        if name == "dt_hour":
            source = self.translate_expr(expr.args[0])
            return source.dt.hour()
        if name == "dt_minute":
            source = self.translate_expr(expr.args[0])
            return source.dt.minute()
        if name == "dt_second":
            source = self.translate_expr(expr.args[0])
            return source.dt.second()
        if name == "dt_truncate":
            source = self.translate_expr(expr.args[0])
            return source.dt.truncate(expr.args[1])

        # Null/NaN handling
        if name == "fill_null":
            source = self.translate_expr(expr.args[0])
            fill_val = self.translate_expr(expr.args[1])
            return source.fill_null(fill_val)
        if name == "fill_nan":
            source = self.translate_expr(expr.args[0])
            fill_val = self.translate_expr(expr.args[1])
            return source.fill_nan(fill_val)
        if name == "assert_non_null":
            return self.translate_expr(expr.args[0])

        # Cast
        if name == "cast":
            source = self.translate_expr(expr.args[0])
            target_dtype = map_colnade_dtype(expr.kwargs["dtype"])
            return source.cast(target_dtype)

        # Window function
        if name == "over":
            source = self.translate_expr(expr.args[0])
            partition_cols = [self.translate_expr(a) for a in expr.args[1:]]
            return source.over(partition_cols)

        msg = f"Unsupported FunctionCall: {name}"
        raise ValueError(msg)

    def _translate_list_op(self, expr: ListOp[Any]) -> pl.Expr:
        """Translate a ListOp node to Polars."""
        list_expr = self.translate_expr(expr.list_expr)
        op = expr.op

        if op == "len":
            return list_expr.list.len()
        if op == "get":
            return list_expr.list.get(expr.args[0])
        if op == "contains":
            return list_expr.list.contains(expr.args[0])
        if op == "sum":
            return list_expr.list.sum()
        if op == "mean":
            return list_expr.list.mean()
        if op == "min":
            return list_expr.list.min()
        if op == "max":
            return list_expr.list.max()

        msg = f"Unsupported ListOp: {op}"
        raise ValueError(msg)

    # --- Execution methods ---

    def filter(self, source: Any, predicate: Expr[Any]) -> Any:
        return source.filter(self.translate_expr(predicate))

    def sort(
        self,
        source: Any,
        by: Sequence[Column[Any] | SortExpr],
        descending: bool,
    ) -> Any:
        sort_cols: list[pl.Expr] = []
        desc_flags: list[bool] = []
        for item in by:
            if isinstance(item, SortExpr):
                sort_cols.append(self.translate_expr(item.expr))
                desc_flags.append(item.descending)
            else:
                # Column instance
                sort_cols.append(pl.col(item.name))
                desc_flags.append(descending)
        return source.sort(sort_cols, descending=desc_flags)

    def limit(self, source: Any, n: int) -> Any:
        return source.limit(n)

    def head(self, source: Any, n: int) -> Any:
        return source.head(n)

    def tail(self, source: Any, n: int) -> Any:
        return source.tail(n)

    def sample(self, source: Any, n: int) -> Any:
        return source.sample(n)

    def unique(self, source: Any, columns: Sequence[Column[Any]]) -> Any:
        return source.unique(subset=[c.name for c in columns])

    def drop_nulls(self, source: Any, columns: Sequence[Column[Any]]) -> Any:
        return source.drop_nulls(subset=[c.name for c in columns])

    def with_columns(self, source: Any, exprs: Sequence[AliasedExpr[Any] | Expr[Any]]) -> Any:
        return source.with_columns([self.translate_expr(e) for e in exprs])

    def select(self, source: Any, columns: Sequence[Column[Any]]) -> Any:
        return source.select([c.name for c in columns])

    def group_by_agg(
        self,
        source: Any,
        keys: Sequence[Column[Any]],
        aggs: Sequence[AliasedExpr[Any]],
    ) -> Any:
        return source.group_by([k.name for k in keys]).agg([self.translate_expr(a) for a in aggs])

    def join(self, left: Any, right: Any, on: Any, how: str) -> Any:
        return left.join(right, left_on=on.left.name, right_on=on.right.name, how=how)

    def cast_schema(self, source: Any, column_mapping: dict[str, str]) -> Any:
        return source.select([pl.col(src).alias(tgt) for tgt, src in column_mapping.items()])

    def lazy(self, source: Any) -> Any:
        return source.lazy()

    def collect(self, source: Any) -> Any:
        return source.collect()

    def validate_schema(self, source: Any, schema: type[Schema]) -> None:
        """Validate that a Polars DataFrame/LazyFrame matches the schema."""
        pl_schema = source.collect_schema() if isinstance(source, pl.LazyFrame) else source.schema
        expected_columns = schema._columns
        actual_names = set(pl_schema.names())

        missing = [n for n in expected_columns if n not in actual_names]
        type_mismatches: dict[str, tuple[str, str]] = {}

        for col_name, col in expected_columns.items():
            if col_name not in actual_names:
                continue
            actual_pl_dtype = pl_schema[col_name]
            expected_pl_dtype = map_colnade_dtype(col.dtype)
            if actual_pl_dtype != expected_pl_dtype:
                actual_colnade = map_polars_dtype(actual_pl_dtype).__name__
                expected_name = (
                    col.dtype.__name__ if hasattr(col.dtype, "__name__") else str(col.dtype)
                )
                type_mismatches[col_name] = (expected_name, actual_colnade)

        if missing or type_mismatches:
            raise SchemaError(
                missing_columns=missing if missing else None,
                type_mismatches=type_mismatches if type_mismatches else None,
            )
