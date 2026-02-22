"""DaskBackend — translates Colnade expression trees and executes operations on Dask DataFrames."""

from __future__ import annotations

import types as _types
from collections.abc import Iterator, Sequence
from typing import Any

import dask.dataframe as dd
import pandas as pd

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
from colnade_pandas.conversion import map_colnade_dtype, map_pandas_dtype

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
# Agg function name mapping (Colnade → Pandas/Dask)
# ---------------------------------------------------------------------------

_AGG_MAP: dict[str, str] = {
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "count": "count",
    "first": "first",
    "last": "last",
}

# ---------------------------------------------------------------------------
# DaskBackend
# ---------------------------------------------------------------------------


class DaskBackend:
    """Colnade backend adapter for Dask.

    Expression translation produces callables ``(df) -> Series | scalar``
    since Dask, like Pandas, has no standalone lazy expression API.  The
    callables build lazy Dask task graphs instead of executing immediately.
    """

    # --- Expression translation ---

    def translate_expr(self, expr: Expr[Any]) -> Any:
        """Recursively translate a Colnade AST node to a callable (df -> result)."""
        if isinstance(expr, AliasedExpr):
            inner_fn = self.translate_expr(expr.expr)
            target_name = expr.target.name
            return (inner_fn, target_name)

        if isinstance(expr, ColumnRef):
            col_name = expr.column.name
            return lambda df, _cn=col_name: df[_cn]

        if isinstance(expr, Literal):
            val = expr.value
            return lambda df, _v=val: _v

        if isinstance(expr, BinOp):
            left_fn = self._ensure_callable(self.translate_expr(expr.left))
            right_fn = self._ensure_callable(self.translate_expr(expr.right))
            method = _BINOP_MAP.get(expr.op)
            if method is None:
                msg = f"Unsupported BinOp operator: {expr.op}"
                raise ValueError(msg)
            return lambda df, _l=left_fn, _r=right_fn, _m=method: getattr(_l(df), _m)(_r(df))

        if isinstance(expr, UnaryOp):
            operand_fn = self._ensure_callable(self.translate_expr(expr.operand))
            if expr.op == "-":
                return lambda df, _o=operand_fn: -_o(df)
            if expr.op == "~":
                return lambda df, _o=operand_fn: ~_o(df)
            if expr.op == "is_null":
                return lambda df, _o=operand_fn: _o(df).isna()
            if expr.op == "is_not_null":
                return lambda df, _o=operand_fn: _o(df).notnull()
            if expr.op == "is_nan":
                return lambda df, _o=operand_fn: _o(df).isna()
            msg = f"Unsupported UnaryOp: {expr.op}"
            raise ValueError(msg)

        if isinstance(expr, Agg):
            source_fn = self._ensure_callable(self.translate_expr(expr.source))
            agg_name = _AGG_MAP.get(expr.agg_type)
            if agg_name is None:
                msg = f"Unsupported aggregation: {expr.agg_type}"
                raise ValueError(msg)
            return (source_fn, agg_name)

        if isinstance(expr, FunctionCall):
            return self._translate_function_call(expr)

        if isinstance(expr, StructFieldAccess):
            struct_fn = self._ensure_callable(self.translate_expr(expr.struct_expr))
            field_name = expr.field.name
            return lambda df, _s=struct_fn, _f=field_name: _s(df).apply(lambda x: x.get(_f))

        if isinstance(expr, ListOp):
            return self._translate_list_op(expr)

        msg = f"Unsupported expression type: {type(expr).__name__}"
        raise TypeError(msg)

    def _ensure_callable(self, translated: Any) -> Any:
        """Unwrap (fn, alias) tuples from AliasedExpr to get the callable."""
        if isinstance(translated, tuple):
            return translated[0]
        return translated

    def _translate_function_call(self, expr: FunctionCall[Any]) -> Any:
        """Translate a FunctionCall node to a Dask callable."""
        name = expr.name

        # String methods
        if name == "str_contains":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            pattern = expr.args[1]
            return lambda df, _s=source_fn, _p=pattern: _s(df).str.contains(_p, regex=False)
        if name == "str_starts_with":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            pattern = expr.args[1]
            return lambda df, _s=source_fn, _p=pattern: _s(df).str.startswith(_p)
        if name == "str_ends_with":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            pattern = expr.args[1]
            return lambda df, _s=source_fn, _p=pattern: _s(df).str.endswith(_p)
        if name == "str_len":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).str.len()
        if name == "str_to_lowercase":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).str.lower()
        if name == "str_to_uppercase":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).str.upper()
        if name == "str_strip":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).str.strip()
        if name == "str_replace":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            old, new = expr.args[1], expr.args[2]
            return lambda df, _s=source_fn, _o=old, _n=new: _s(df).str.replace(_o, _n, regex=False)

        # Temporal methods
        if name == "dt_year":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).dt.year
        if name == "dt_month":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).dt.month
        if name == "dt_day":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).dt.day
        if name == "dt_hour":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).dt.hour
        if name == "dt_minute":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).dt.minute
        if name == "dt_second":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            return lambda df, _s=source_fn: _s(df).dt.second
        if name == "dt_truncate":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            unit = expr.args[1]
            return lambda df, _s=source_fn, _u=unit: _s(df).dt.floor(_u)

        # Null/NaN handling
        if name == "fill_null":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            fill_fn = self._ensure_callable(self.translate_expr(expr.args[1]))
            return lambda df, _s=source_fn, _f=fill_fn: _s(df).fillna(_f(df))
        if name == "fill_nan":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            fill_fn = self._ensure_callable(self.translate_expr(expr.args[1]))
            return lambda df, _s=source_fn, _f=fill_fn: _s(df).fillna(_f(df))
        if name == "assert_non_null":
            return self._ensure_callable(self.translate_expr(expr.args[0]))

        # Cast
        if name == "cast":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            target_dtype = map_colnade_dtype(expr.kwargs["dtype"])
            return lambda df, _s=source_fn, _t=target_dtype: _s(df).astype(_t)

        # Window function (over)
        if name == "over":
            source_fn = self._ensure_callable(self.translate_expr(expr.args[0]))
            partition_names = [self._ensure_callable(self.translate_expr(a)) for a in expr.args[1:]]
            return lambda df, _s=source_fn, _p=partition_names: df.groupby(
                [p(df).name for p in _p]
            )[_s(df).name].transform(lambda x: x)

        msg = f"Unsupported FunctionCall: {name}"
        raise ValueError(msg)

    def _translate_list_op(self, expr: ListOp[Any]) -> Any:
        """Translate a ListOp node to a Dask callable."""
        list_fn = self._ensure_callable(self.translate_expr(expr.list_expr))
        op = expr.op

        if op == "len":
            return lambda df, _l=list_fn: _l(df).apply(len, meta=(_l(df).name, "int64"))
        if op == "get":
            idx = expr.args[0]
            return lambda df, _l=list_fn, _i=idx: _l(df).apply(
                lambda x: x[_i], meta=(_l(df).name, "object")
            )
        if op == "contains":
            val = expr.args[0]
            return lambda df, _l=list_fn, _v=val: _l(df).apply(
                lambda x: _v in x, meta=(_l(df).name, "bool")
            )
        if op == "sum":
            return lambda df, _l=list_fn: _l(df).apply(sum, meta=(_l(df).name, "float64"))
        if op == "mean":
            return lambda df, _l=list_fn: _l(df).apply(
                lambda x: sum(x) / len(x) if len(x) > 0 else None,
                meta=(_l(df).name, "float64"),
            )
        if op == "min":
            return lambda df, _l=list_fn: _l(df).apply(min, meta=(_l(df).name, "object"))
        if op == "max":
            return lambda df, _l=list_fn: _l(df).apply(max, meta=(_l(df).name, "object"))

        msg = f"Unsupported ListOp: {op}"
        raise ValueError(msg)

    # --- Execution methods ---

    def filter(self, source: Any, predicate: Expr[Any]) -> Any:
        pred_fn = self._ensure_callable(self.translate_expr(predicate))
        mask = pred_fn(source)
        return source.loc[mask].reset_index(drop=True)

    def sort(
        self,
        source: Any,
        by: Sequence[Column[Any] | SortExpr],
        descending: bool,
    ) -> Any:
        col_names: list[str] = []
        ascending: list[bool] = []
        for item in by:
            if isinstance(item, SortExpr):
                col_names.append(item.expr.column.name)
                ascending.append(not item.descending)
            else:
                col_names.append(item.name)
                ascending.append(not descending)
        return source.sort_values(by=col_names, ascending=ascending).reset_index(drop=True)

    def limit(self, source: Any, n: int) -> Any:
        return source.head(n, npartitions=-1, compute=False)

    def head(self, source: Any, n: int) -> Any:
        return source.head(n, npartitions=-1, compute=False)

    def tail(self, source: Any, n: int) -> Any:
        # Dask tail() returns a Pandas DF — re-wrap in Dask
        return dd.from_pandas(source.tail(n), npartitions=1)

    def sample(self, source: Any, n: int) -> Any:
        # Dask doesn't support sample(n=...), only sample(frac=...)
        # Compute to Pandas, sample, and re-wrap
        computed = source.compute()
        sampled = computed.sample(n).reset_index(drop=True)
        return dd.from_pandas(sampled, npartitions=1)

    def unique(self, source: Any, columns: Sequence[Column[Any]]) -> Any:
        return source.drop_duplicates(subset=[c.name for c in columns]).reset_index(drop=True)

    def drop_nulls(self, source: Any, columns: Sequence[Column[Any]]) -> Any:
        return source.dropna(subset=[c.name for c in columns]).reset_index(drop=True)

    def with_columns(self, source: Any, exprs: Sequence[AliasedExpr[Any] | Expr[Any]]) -> Any:
        result = source
        for expr in exprs:
            translated = self.translate_expr(expr)
            if isinstance(translated, tuple):
                fn, alias = translated
                result = result.assign(**{alias: fn(result)})
            else:
                msg = "with_columns requires aliased expressions"
                raise ValueError(msg)
        return result

    def select(self, source: Any, columns: Sequence[Column[Any]]) -> Any:
        return source[[c.name for c in columns]]

    def group_by_agg(
        self,
        source: Any,
        keys: Sequence[Column[Any]],
        aggs: Sequence[AliasedExpr[Any]],
    ) -> Any:
        key_names = [k.name for k in keys]
        agg_dict: dict[str, str] = {}
        rename_map: dict[str, str] = {}

        for agg_expr in aggs:
            translated = self.translate_expr(agg_expr)
            if not isinstance(translated, tuple):
                msg = "group_by_agg requires aliased aggregation expressions"
                raise ValueError(msg)

            inner, alias = translated
            if isinstance(inner, tuple):
                source_fn, agg_name = inner
                col_name = self._extract_col_name(source_fn, source)
                agg_dict[col_name] = agg_name
                rename_map[col_name] = alias
            else:
                msg = "group_by_agg requires aggregation expressions (e.g., .sum(), .mean())"
                raise ValueError(msg)

        grouped = source.groupby(key_names).agg(agg_dict).reset_index()
        for old_name, new_name in rename_map.items():
            if old_name != new_name and old_name not in key_names:
                grouped = grouped.rename(columns={old_name: new_name})
        return grouped

    def agg(self, source: Any, aggs: Sequence[AliasedExpr[Any]]) -> Any:
        result: dict[str, Any] = {}
        for agg_expr in aggs:
            translated = self.translate_expr(agg_expr)
            inner, alias = translated
            source_fn, agg_name = inner
            col_name = self._extract_col_name(source_fn, source)
            val = getattr(source[col_name], agg_name)()
            result[alias] = val.compute() if hasattr(val, "compute") else val
        return dd.from_pandas(pd.DataFrame([result]), npartitions=1)

    def _extract_col_name(self, fn: Any, df: Any) -> str:
        """Extract column name from a translated ColumnRef function."""
        series = fn(df)
        if hasattr(series, "name"):
            return series.name
        msg = "Cannot extract column name from expression"
        raise ValueError(msg)

    @staticmethod
    def _dtypes_compatible(actual: Any, expected: Any) -> bool:
        """Compare dtypes accounting for storage backend variations.

        Dask may use different storage backends (e.g., pyarrow vs python)
        for the same logical dtype, so strict equality can fail.
        """
        if actual == expected:
            return True
        # StringDtype with different storage backends (python vs pyarrow)
        return isinstance(actual, pd.StringDtype) and isinstance(expected, pd.StringDtype)

    def join(self, left: Any, right: Any, on: Any, how: str) -> Any:
        return left.merge(right, left_on=on.left.name, right_on=on.right.name, how=how)

    def cast_schema(self, source: Any, column_mapping: dict[str, str]) -> Any:
        rename_map = {src: tgt for tgt, src in column_mapping.items()}
        result = source.rename(columns=rename_map)
        return result[list(column_mapping.keys())]

    def lazy(self, source: Any) -> Any:
        # Dask is inherently lazy — passthrough
        return source

    def collect(self, source: Any) -> Any:
        # Materialize the Dask task graph and re-wrap so DaskBackend
        # can continue operating on the result.
        return dd.from_pandas(source.compute(), npartitions=1)

    def validate_schema(self, source: Any, schema: type[Schema]) -> None:
        """Validate that a Dask DataFrame matches the schema."""
        expected_columns = schema._columns
        actual_names = set(source.columns)

        missing = [n for n in expected_columns if n not in actual_names]
        type_mismatches: dict[str, tuple[str, str]] = {}

        for col_name, col in expected_columns.items():
            if col_name not in actual_names:
                continue
            actual_pd_dtype = source[col_name].dtype
            expected_pd_dtype = map_colnade_dtype(col.dtype)
            if not self._dtypes_compatible(actual_pd_dtype, expected_pd_dtype):
                try:
                    actual_colnade = map_pandas_dtype(actual_pd_dtype).__name__
                except TypeError:
                    actual_colnade = str(actual_pd_dtype)
                expected_name = (
                    col.dtype.__name__ if hasattr(col.dtype, "__name__") else str(col.dtype)
                )
                type_mismatches[col_name] = (expected_name, actual_colnade)

        # Null checks require computation in Dask
        null_violations: list[str] = []
        for col_name, col in expected_columns.items():
            if col_name not in actual_names:
                continue
            if isinstance(col.dtype, _types.UnionType):
                continue
            if source[col_name].isna().any().compute():
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

        # Materialize Dask DataFrame for value checks
        pdf = source.compute() if hasattr(source, "compute") else source
        violations: list[ValueViolation] = []

        for col_name, field_info in constraints.items():
            if col_name not in pdf.columns:
                continue
            series = pdf[col_name].dropna()

            if field_info.ge is not None:
                mask = series < field_info.ge
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].head(5).tolist()
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
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].head(5).tolist()
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
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].head(5).tolist()
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
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].head(5).tolist()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"lt={field_info.lt!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.min_length is not None:
                lengths = series.str.len()
                mask = lengths < field_info.min_length
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].head(5).tolist()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"min_length={field_info.min_length}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.max_length is not None:
                lengths = series.str.len()
                mask = lengths > field_info.max_length
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].head(5).tolist()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"max_length={field_info.max_length}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.pattern is not None:
                matches = series.str.contains(field_info.pattern, regex=True, na=False)
                mask = ~matches
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].head(5).tolist()
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint=f"pattern={field_info.pattern!r}",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.unique:
                dup_mask = series.duplicated(keep=False)
                count = int(dup_mask.sum())
                if count > 0:
                    samples = series[dup_mask].unique().tolist()[:5]
                    violations.append(
                        ValueViolation(
                            column=col_name,
                            constraint="unique",
                            got_count=count,
                            sample_values=samples,
                        )
                    )

            if field_info.isin is not None:
                mask = ~series.isin(field_info.isin)
                count = int(mask.sum())
                if count > 0:
                    samples = series[mask].unique().tolist()[:5]
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
            pd_mask = self.translate_expr(expr)
            violation_count = int((~pd_mask(pdf)).sum())
            if violation_count > 0:
                violations.append(
                    ValueViolation(
                        column="(cross-column)",
                        constraint=f"schema_check:{check.name}",
                        got_count=violation_count,
                        sample_values=[],
                    )
                )

        if violations:
            raise SchemaError(value_violations=violations)

    # --- Introspection ---

    def row_count(self, source: Any) -> int:
        return len(source)

    def iter_row_dicts(self, source: Any) -> Iterator[dict[str, Any]]:
        return source.compute().to_dict(orient="records")

    # --- Arrow boundary ---

    def to_arrow_batches(
        self,
        source: Any,
        batch_size: int | None,
    ) -> Iterator[Any]:
        """Convert a Dask DataFrame to an iterator of Arrow RecordBatches."""
        import pyarrow as pa

        # Iterate partitions for memory-efficient conversion
        for partition in source.partitions:
            pdf: pd.DataFrame = partition.compute()
            table: pa.Table = pa.Table.from_pandas(pdf)
            if batch_size is not None:
                yield from table.to_batches(max_chunksize=batch_size)
            else:
                yield from table.to_batches()

    def from_arrow_batches(
        self,
        batches: Iterator[Any],
        schema: type[Any],
    ) -> Any:
        """Reconstruct a Dask DataFrame from Arrow RecordBatches."""
        import pyarrow as pa

        batch_list = list(batches)
        if not batch_list:
            return dd.from_pandas(pd.DataFrame(), npartitions=1)
        table = pa.Table.from_batches(batch_list)
        pdf = table.to_pandas()
        return dd.from_pandas(pdf, npartitions=1)

    def from_dict(
        self,
        data: dict[str, Sequence[Any]],
        schema: type[Schema],
    ) -> Any:
        """Create a Dask DataFrame from a columnar dict with schema-driven dtypes."""
        from colnade_pandas.conversion import map_colnade_dtype

        pd_schema = {name: map_colnade_dtype(col.dtype) for name, col in schema._columns.items()}
        pdf = pd.DataFrame(data).astype(pd_schema)
        return dd.from_pandas(pdf, npartitions=1)
