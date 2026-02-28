"""PolarsBackend — translates Colnade expression trees and executes operations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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
    WhenThenOtherwise,
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

        if isinstance(expr, WhenThenOtherwise):
            cond, val = expr.cases[0]
            result = pl.when(self.translate_expr(cond)).then(self.translate_expr(val))
            for cond, val in expr.cases[1:]:
                result = result.when(self.translate_expr(cond)).then(self.translate_expr(val))
            return result.otherwise(self.translate_expr(expr.otherwise_expr))

        msg = f"Unsupported expression type: {type(expr).__name__}"
        raise TypeError(msg)

    def _translate_function_call(self, expr: FunctionCall[Any]) -> pl.Expr:
        """Translate a FunctionCall node to Polars."""
        name = expr.name

        # String methods
        if name == "str_contains":
            source = self.translate_expr(expr.args[0])
            return source.str.contains(expr.args[1], literal=True)
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
            source = self.translate_expr(expr.args[0])

            def _check_nulls(s: Any) -> Any:
                if s.null_count() > 0:
                    raise ValueError(
                        f"assert_non_null failed: column contains {s.null_count()} null values"
                    )
                return s

            return source.map_batches(_check_nulls)

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
        subset = [c.name for c in columns] or None
        return source.unique(subset=subset)

    def drop_nulls(self, source: Any, columns: Sequence[Column[Any]]) -> Any:
        return source.drop_nulls(subset=[c.name for c in columns])

    def with_columns(self, source: Any, exprs: Sequence[AliasedExpr[Any] | Expr[Any]]) -> Any:
        return source.with_columns([self.translate_expr(e) for e in exprs])

    def select(self, source: Any, columns: Sequence[Column[Any] | str]) -> Any:
        return source.select([c.name if hasattr(c, "name") else c for c in columns])

    def group_by_agg(
        self,
        source: Any,
        keys: Sequence[Column[Any]],
        aggs: Sequence[AliasedExpr[Any]],
    ) -> Any:
        return source.group_by([k.name for k in keys]).agg([self.translate_expr(a) for a in aggs])

    def agg(self, source: Any, aggs: Sequence[AliasedExpr[Any]]) -> Any:
        return source.select([self.translate_expr(a) for a in aggs])

    def join(self, left: Any, right: Any, on: Any, how: str) -> Any:
        return left.join(right, left_on=on.left.name, right_on=on.right.name, how=how)

    def concat(self, sources: Sequence[Any]) -> Any:
        return pl.concat(list(sources))

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

        # Check nullability — non-nullable columns should have no nulls
        null_violations: list[str] = []
        if isinstance(source, pl.DataFrame):
            import types

            for col_name, col in expected_columns.items():
                if col_name not in actual_names:
                    continue
                if isinstance(col.dtype, types.UnionType):
                    continue
                if source[col_name].null_count() > 0:
                    null_violations.append(col_name)

        if missing or type_mismatches or null_violations:
            raise SchemaError(
                missing_columns=missing if missing else None,
                type_mismatches=type_mismatches if type_mismatches else None,
                null_violations=null_violations if null_violations else None,
            )

    def validate_field_constraints(self, source: Any, schema: type[Schema]) -> None:
        """Validate value-level constraints (Field(), @schema_check) on data."""
        from colnade.constraints import ValueViolation, get_column_constraints, get_schema_checks

        constraints = get_column_constraints(schema)
        checks = get_schema_checks(schema)
        if not constraints and not checks:
            return

        # Materialize LazyFrame for value checks
        df = source.collect() if isinstance(source, pl.LazyFrame) else source
        violations: list[ValueViolation] = []

        for col_name, field_info in constraints.items():
            if col_name not in df.columns:
                continue
            series = df[col_name].drop_nulls()

            if field_info.ge is not None:
                mask = series < field_info.ge
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"ge={field_info.ge!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.gt is not None:
                mask = series <= field_info.gt
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"gt={field_info.gt!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.le is not None:
                mask = series > field_info.le
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"le={field_info.le!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.lt is not None:
                mask = series >= field_info.lt
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"lt={field_info.lt!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.min_length is not None:
                lengths = series.str.len_chars()
                mask = lengths < field_info.min_length
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"min_length={field_info.min_length}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.max_length is not None:
                lengths = series.str.len_chars()
                mask = lengths > field_info.max_length
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"max_length={field_info.max_length}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.pattern is not None:
                matches = series.str.contains(field_info.pattern)
                mask = ~matches
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"pattern={field_info.pattern!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.unique:
                dup_mask = series.is_duplicated()
                count = dup_mask.sum()
                if count > 0:
                    samples = series.filter(dup_mask).unique().head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint="unique",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.isin is not None:
                allowed = list(field_info.isin)
                mask = ~series.is_in(allowed)
                count = mask.sum()
                if count > 0:
                    samples = series.filter(mask).unique().head(5).to_list()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"isin={list(field_info.isin)!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

        # Cross-column schema checks
        for check in checks:
            expr = check.fn(schema)
            pl_expr = self.translate_expr(expr)
            violation_mask = ~pl_expr
            count = df.select(violation_mask.alias("__check")).get_column("__check").sum()
            if count > 0:
                violations.append(
                    ValueViolation(
                        column="(cross-column)",
                        constraint=f"schema_check:{check.name}",
                        got_count=count,
                        sample_values=[],
                    )
                )

        if violations:
            raise SchemaError(value_violations=violations)

    # --- Row access ---

    def row_count(self, source: Any) -> int:
        return source.height

    def iter_row_dicts(self, source: Any) -> Iterator[dict[str, Any]]:
        return source.iter_rows(named=True)

    def item(self, source: Any, column: str | None = None) -> Any:
        import polars as pl

        if isinstance(source, pl.LazyFrame):
            source = source.collect()
        if column is not None:
            if source.shape[0] != 1:
                msg = f"item() requires exactly 1 row, got {source.shape[0]}"
                raise ValueError(msg)
            return source[column].item()
        if source.shape != (1, 1):
            msg = f"item() requires a 1\u00d71 DataFrame, got shape {source.shape}"
            raise ValueError(msg)
        return source.item()

    # --- Arrow boundary ---

    def to_arrow_batches(
        self,
        source: Any,
        batch_size: int | None,
    ) -> Iterator[Any]:
        """Convert a Polars DataFrame to an iterator of Arrow RecordBatches."""
        import pyarrow as pa  # noqa: F811

        if hasattr(source, "collect"):
            source = source.collect()
        table: pa.Table = source.to_arrow()
        if batch_size is not None:
            yield from table.to_batches(max_chunksize=batch_size)
        else:
            yield from table.to_batches()

    def from_arrow_batches(
        self,
        batches: Iterator[Any],
        schema: type[Any],
    ) -> Any:
        """Reconstruct a Polars DataFrame from Arrow RecordBatches."""
        import pyarrow as pa  # noqa: F811

        batch_list = list(batches)
        if not batch_list:
            return pl.DataFrame()
        table = pa.Table.from_batches(batch_list)
        return pl.from_arrow(table)

    def from_dict(
        self,
        data: dict[str, Sequence[Any]],
        schema: type[Schema],
    ) -> Any:
        """Create a Polars DataFrame from a columnar dict with schema-driven dtypes."""
        from colnade_polars.conversion import map_colnade_dtype

        pl_schema = {name: map_colnade_dtype(col.dtype) for name, col in schema._columns.items()}
        return pl.DataFrame(data, schema=pl_schema)
