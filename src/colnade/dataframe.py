"""DataFrame[S], LazyFrame[S], GroupBy, and untyped escape hatches.

Typed DataFrame interfaces parameterized by Schema. All data operations
require a backend adapter (``_backend``); calling them without one raises
``RuntimeError``.  Construction, repr, ``to_native()``, ``untyped()``, and
``validate()`` still work without a backend.

Limitation: Column[DType] has a single type parameter (no schema binding),
so methods like select(), sort(), group_by() accept Column[Any] and cannot
statically enforce that columns belong to the DataFrame's schema S. Schema
enforcement at the column level requires a second type parameter on Column,
which was dropped due to Python 3.10 lacking TypeVar defaults (PEP 696).
See AGENTS.md "Column[DType] Annotation Pattern" for details.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from colnade.schema import S2, S3, Column, S, Schema, SchemaError

R = TypeVar("R")

if TYPE_CHECKING:
    from typing import Literal

    from colnade._protocols import BackendProtocol
    from colnade.arrow import ArrowBatch
    from colnade.dtypes import Bool
    from colnade.expr import AliasedExpr, Expr, JoinCondition, SortExpr


# ---------------------------------------------------------------------------
# Backend guard
# ---------------------------------------------------------------------------

_NO_BACKEND_MSG = (
    "This operation requires a backend adapter. "
    "Construct the frame via a backend (e.g. read_parquet()) "
    "or pass _backend=... to the constructor."
)


def _require_backend(backend: BackendProtocol | None) -> BackendProtocol:
    """Return *backend* or raise ``RuntimeError`` if ``None``."""
    if backend is None:
        raise RuntimeError(_NO_BACKEND_MSG)
    return backend


# ---------------------------------------------------------------------------
# cast_schema resolution helper
# ---------------------------------------------------------------------------


def _resolve_mapping(
    target_schema: type[Schema],
    source_columns: dict[str, Column[Any]],
    mapping: dict[Column[Any], Column[Any]] | None,
    extra: Literal["drop", "forbid"],
    ambiguous_names: set[str] | None = None,
) -> dict[str, str]:
    """Resolve target→source column name mapping.

    Returns ``{target_name: source_name}``.
    Raises :class:`SchemaError` on missing or (when ``extra="forbid"``) extra columns.

    Resolution precedence per target column:
    1. Explicit ``mapping`` dict
    2. Target column's ``_mapped_from`` attribute
    3. Name matching against source columns (skipped for ambiguous names)
    """
    result: dict[str, str] = {}
    explicit = mapping or {}
    ambiguous = ambiguous_names or set()
    target_columns: dict[str, Column[Any]] = target_schema._columns

    for target_name, target_col in target_columns.items():
        # 1. Explicit mapping
        if target_col in explicit:
            result[target_name] = explicit[target_col].name
            continue
        # 2. mapped_from
        if target_col._mapped_from is not None:
            result[target_name] = target_col._mapped_from.name
            continue
        # 3. Name matching (not for ambiguous names in joined schemas)
        if target_name in source_columns and target_name not in ambiguous:
            result[target_name] = target_name
            continue

    missing = [name for name in target_columns if name not in result]
    if missing:
        raise SchemaError(missing_columns=missing)

    if extra == "forbid":
        used = set(result.values())
        extra_cols = sorted(set(source_columns) - used)
        if extra_cols:
            raise SchemaError(extra_columns=extra_cols)

    return result


# ---------------------------------------------------------------------------
# DataFrame[S] — typed, materialized DataFrame
# ---------------------------------------------------------------------------


class DataFrame(Generic[S]):
    """A typed, materialized DataFrame parameterized by a Schema.

    Schema-preserving operations (filter, sort, limit, etc.) return
    ``DataFrame[S]``. Schema-transforming operations (select, group_by+agg)
    return ``DataFrame[Any]`` and require ``cast_schema()`` to bind to a
    named output schema.
    """

    __slots__ = ("_data", "_schema", "_backend")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema: type[Any] | None = None,
        _backend: BackendProtocol | None = None,
    ) -> None:
        self._data = _data
        self._schema = _schema
        self._backend = _backend

    def __repr__(self) -> str:
        schema_name = self._schema.__name__ if self._schema else "Any"
        header = f"DataFrame[{schema_name}]"
        if self._data is not None and hasattr(self._data, "__repr__"):
            return f"{header}\n{self._data!r}"
        return header

    def _repr_html_(self) -> str | None:
        """Rich HTML representation for Jupyter notebooks."""
        schema_name = self._schema.__name__ if self._schema else "Any"
        header = f"<b>DataFrame[{schema_name}]</b>"
        if self._data is not None and hasattr(self._data, "_repr_html_"):
            return f"{header}\n{self._data._repr_html_()}"
        return None

    def to_native(self) -> Any:
        """Return the underlying backend-native data object (e.g. pl.DataFrame)."""
        return self._data

    # --- Introspection ---

    @property
    def height(self) -> int:
        """Return the number of rows."""
        return _require_backend(self._backend).row_count(self._data)

    def __len__(self) -> int:
        """Return the number of rows."""
        return self.height

    @property
    def width(self) -> int:
        """Return the number of columns.

        Raises ``TypeError`` on ``DataFrame[Any]`` (schema erased).
        Use ``cast_schema()`` first to bind to a named schema.
        """
        if self._schema is None:
            msg = (
                "width is not available on DataFrame[Any] (schema erased). "
                "Use cast_schema() first to bind to a named schema."
            )
            raise TypeError(msg)
        return len(self._schema._columns)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``."""
        return (self.height, self.width)

    def is_empty(self) -> bool:
        """Return ``True`` if the DataFrame has zero rows."""
        return self.height == 0

    def iter_rows_as(self, row_type: type[R]) -> Iterator[R]:
        """Iterate rows, constructing *row_type* instances via ``row_type(**row_dict)``.

        Works with ``Schema.Row`` (frozen dataclass), ``dict``, plain
        dataclasses, ``NamedTuple``, Pydantic models, or any callable
        accepting ``**kwargs``.
        """
        backend = _require_backend(self._backend)
        for d in backend.iter_row_dicts(self._data):
            yield row_type(**d)

    # --- Schema-preserving operations (return DataFrame[S]) ---

    def filter(self, predicate: Expr[Bool]) -> DataFrame[S]:
        """Filter rows by a boolean expression."""
        data = _require_backend(self._backend).filter(self._data, predicate)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def sort(self, *columns: Column[Any] | SortExpr, descending: bool = False) -> DataFrame[S]:
        """Sort rows by columns or sort expressions."""
        data = _require_backend(self._backend).sort(self._data, columns, descending)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def limit(self, n: int) -> DataFrame[S]:
        """Limit to the first n rows."""
        data = _require_backend(self._backend).limit(self._data, n)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def head(self, n: int = 5) -> DataFrame[S]:
        """Return the first n rows (materialized only)."""
        data = _require_backend(self._backend).head(self._data, n)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def tail(self, n: int = 5) -> DataFrame[S]:
        """Return the last n rows (materialized only)."""
        data = _require_backend(self._backend).tail(self._data, n)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def sample(self, n: int) -> DataFrame[S]:
        """Return a random sample of n rows (materialized only)."""
        data = _require_backend(self._backend).sample(self._data, n)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def unique(self, *columns: Column[Any]) -> DataFrame[S]:
        """Remove duplicate rows based on the given columns."""
        data = _require_backend(self._backend).unique(self._data, columns)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def drop_nulls(self, *columns: Column[Any]) -> DataFrame[S]:
        """Drop rows with null values in the given columns."""
        data = _require_backend(self._backend).drop_nulls(self._data, columns)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> DataFrame[S]:
        """Add or overwrite columns. Returns DataFrame[S] (optimistic)."""
        data = _require_backend(self._backend).with_columns(self._data, exprs)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    # --- Schema-transforming operations (return DataFrame[Any]) ---

    @overload
    def select(self, c1: Column[Any], /) -> DataFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], /) -> DataFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], c3: Column[Any], /) -> DataFrame[Any]: ...
    @overload
    def select(
        self, c1: Column[Any], c2: Column[Any], c3: Column[Any], c4: Column[Any], /
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        c10: Column[Any],
        /,
    ) -> DataFrame[Any]: ...

    def select(self, *columns: Column[Any]) -> DataFrame[Any]:
        """Select columns. Returns DataFrame[Any] — use cast_schema() to bind."""
        data = _require_backend(self._backend).select(self._data, columns)
        return DataFrame(_data=data, _schema=None, _backend=self._backend)

    # --- GroupBy ---

    def group_by(self, *keys: Column[Any]) -> GroupBy[S]:
        """Group by columns for aggregation."""
        return GroupBy(_data=self._data, _schema=self._schema, _keys=keys, _backend=self._backend)

    # --- Join ---

    def join(
        self,
        other: DataFrame[S2],
        on: JoinCondition,
        how: Literal["inner", "left", "outer", "cross"] = "inner",
    ) -> JoinedDataFrame[S, S2]:
        """Join with another DataFrame on a JoinCondition."""
        data = _require_backend(self._backend).join(self._data, other._data, on, how)
        return JoinedDataFrame(
            _data=data,
            _schema_left=self._schema,
            _schema_right=other._schema,
            _backend=self._backend,
        )

    # --- Schema transition ---

    def cast_schema(
        self,
        schema: type[S3],
        mapping: dict[Column[Any], Column[Any]] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> DataFrame[S3]:
        """Bind to a new schema via mapping resolution."""
        backend = _require_backend(self._backend)
        data = self._data
        if self._schema is not None:
            name_map = _resolve_mapping(schema, self._schema._columns, mapping, extra)
            data = backend.cast_schema(data, name_map)
        return DataFrame(_data=data, _schema=schema, _backend=self._backend)

    # --- Conversion ---

    def lazy(self) -> LazyFrame[S]:
        """Convert to a lazy query plan."""
        data = _require_backend(self._backend).lazy(self._data)
        return LazyFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def untyped(self) -> UntypedDataFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedDataFrame(_data=self._data, _backend=self._backend)

    def with_raw(self, fn: Callable[[Any], Any]) -> DataFrame[S]:
        """Apply a function to the raw engine DataFrame and re-wrap.

        The function receives the underlying engine DataFrame (e.g.
        ``pl.DataFrame``, ``pd.DataFrame``) and must return the same type.
        The result is wrapped back into ``DataFrame[S]`` with the same
        schema and backend.  If validation is enabled, the result is
        validated before returning.

        Use this instead of ``untyped()`` when you need a bounded escape
        hatch — like Rust's ``unsafe`` block.
        """
        from colnade.validation import is_validation_enabled

        _require_backend(self._backend)
        result = fn(self._data)
        df: DataFrame[S] = DataFrame(
            _data=result, _schema=self._schema, _backend=self._backend
        )
        if is_validation_enabled():
            df.validate()
        return df

    # --- Validation ---

    def validate(self) -> DataFrame[S]:
        """Validate that the data conforms to the schema."""
        if self._backend and self._schema:
            self._backend.validate_schema(self._data, self._schema)
        return self

    def to_batches(self, batch_size: int | None = None) -> Iterator[ArrowBatch[S]]:
        """Convert to an iterator of typed Arrow batches.

        Delegates to the backend's ``to_arrow_batches()`` method, wrapping
        each raw ``pa.RecordBatch`` in an ``ArrowBatch[S]`` to preserve
        schema type information across the boundary.
        """
        from colnade.arrow import ArrowBatch

        backend = _require_backend(self._backend)
        for raw_batch in backend.to_arrow_batches(self._data, batch_size):
            yield ArrowBatch(_batch=raw_batch, _schema=self._schema)

    @classmethod
    def from_batches(
        cls,
        batches: Iterator[ArrowBatch[S]],
        schema: type[S],
        backend: BackendProtocol,
    ) -> DataFrame[S]:
        """Create a DataFrame from an iterator of typed Arrow batches.

        Unwraps each ``ArrowBatch[S]`` to its raw ``pa.RecordBatch`` and
        delegates to the backend's ``from_arrow_batches()`` method.
        """
        from colnade.validation import is_validation_enabled

        raw_batches = (batch.to_pyarrow() for batch in batches)
        data = backend.from_arrow_batches(raw_batches, schema)
        df: DataFrame[S] = DataFrame(_data=data, _schema=schema, _backend=backend)
        if is_validation_enabled():
            df.validate()
        return df


# ---------------------------------------------------------------------------
# LazyFrame[S] — typed, lazy query plan
# ---------------------------------------------------------------------------


class LazyFrame(Generic[S]):
    """A typed, lazy query plan parameterized by a Schema.

    Same operations as DataFrame except: no head(), tail(), sample(),
    to_batches() (materialized-only ops). Use collect() to materialize.
    """

    __slots__ = ("_data", "_schema", "_backend")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema: type[Any] | None = None,
        _backend: BackendProtocol | None = None,
    ) -> None:
        self._data = _data
        self._schema = _schema
        self._backend = _backend

    def __repr__(self) -> str:
        schema_name = self._schema.__name__ if self._schema else "Any"
        header = f"LazyFrame[{schema_name}]"
        if self._data is not None and hasattr(self._data, "__repr__"):
            return f"{header}\n{self._data!r}"
        return header

    def _repr_html_(self) -> str | None:
        """Rich HTML representation for Jupyter notebooks."""
        schema_name = self._schema.__name__ if self._schema else "Any"
        header = f"<b>LazyFrame[{schema_name}]</b>"
        if self._data is not None and hasattr(self._data, "_repr_html_"):
            return f"{header}\n{self._data._repr_html_()}"
        return None

    def to_native(self) -> Any:
        """Return the underlying backend-native data object (e.g. pl.LazyFrame)."""
        return self._data

    # --- Introspection ---

    @property
    def width(self) -> int:
        """Return the number of columns.

        Derivable from the schema without materializing. Raises ``TypeError``
        on ``LazyFrame[Any]`` (schema erased).
        """
        if self._schema is None:
            msg = (
                "width is not available on LazyFrame[Any] (schema erased). "
                "Use cast_schema() first to bind to a named schema."
            )
            raise TypeError(msg)
        return len(self._schema._columns)

    # --- Schema-preserving operations (return LazyFrame[S]) ---

    def filter(self, predicate: Expr[Bool]) -> LazyFrame[S]:
        """Filter rows by a boolean expression."""
        data = _require_backend(self._backend).filter(self._data, predicate)
        return LazyFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def sort(self, *columns: Column[Any] | SortExpr, descending: bool = False) -> LazyFrame[S]:
        """Sort rows by columns or sort expressions."""
        data = _require_backend(self._backend).sort(self._data, columns, descending)
        return LazyFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def limit(self, n: int) -> LazyFrame[S]:
        """Limit to the first n rows."""
        data = _require_backend(self._backend).limit(self._data, n)
        return LazyFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def unique(self, *columns: Column[Any]) -> LazyFrame[S]:
        """Remove duplicate rows based on the given columns."""
        data = _require_backend(self._backend).unique(self._data, columns)
        return LazyFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def drop_nulls(self, *columns: Column[Any]) -> LazyFrame[S]:
        """Drop rows with null values in the given columns."""
        data = _require_backend(self._backend).drop_nulls(self._data, columns)
        return LazyFrame(_data=data, _schema=self._schema, _backend=self._backend)

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> LazyFrame[S]:
        """Add or overwrite columns. Returns LazyFrame[S] (optimistic)."""
        data = _require_backend(self._backend).with_columns(self._data, exprs)
        return LazyFrame(_data=data, _schema=self._schema, _backend=self._backend)

    # --- Schema-transforming operations (return LazyFrame[Any]) ---

    @overload
    def select(self, c1: Column[Any], /) -> LazyFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], /) -> LazyFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], c3: Column[Any], /) -> LazyFrame[Any]: ...
    @overload
    def select(
        self, c1: Column[Any], c2: Column[Any], c3: Column[Any], c4: Column[Any], /
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        c10: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...

    def select(self, *columns: Column[Any]) -> LazyFrame[Any]:
        """Select columns. Returns LazyFrame[Any] — use cast_schema() to bind."""
        data = _require_backend(self._backend).select(self._data, columns)
        return LazyFrame(_data=data, _schema=None, _backend=self._backend)

    # --- GroupBy ---

    def group_by(self, *keys: Column[Any]) -> LazyGroupBy[S]:
        """Group by columns for aggregation."""
        return LazyGroupBy(
            _data=self._data, _schema=self._schema, _keys=keys, _backend=self._backend
        )

    # --- Join ---

    def join(
        self,
        other: LazyFrame[S2],
        on: JoinCondition,
        how: Literal["inner", "left", "outer", "cross"] = "inner",
    ) -> JoinedLazyFrame[S, S2]:
        """Join with another LazyFrame on a JoinCondition."""
        data = _require_backend(self._backend).join(self._data, other._data, on, how)
        return JoinedLazyFrame(
            _data=data,
            _schema_left=self._schema,
            _schema_right=other._schema,
            _backend=self._backend,
        )

    # --- Schema transition ---

    def cast_schema(
        self,
        schema: type[S3],
        mapping: dict[Column[Any], Column[Any]] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> LazyFrame[S3]:
        """Bind to a new schema via mapping resolution."""
        backend = _require_backend(self._backend)
        data = self._data
        if self._schema is not None:
            name_map = _resolve_mapping(schema, self._schema._columns, mapping, extra)
            data = backend.cast_schema(data, name_map)
        return LazyFrame(_data=data, _schema=schema, _backend=self._backend)

    # --- Materialization ---

    def collect(self) -> DataFrame[S]:
        """Materialize the lazy query plan into a DataFrame."""
        data = _require_backend(self._backend).collect(self._data)
        return DataFrame(_data=data, _schema=self._schema, _backend=self._backend)

    # --- Conversion ---

    def untyped(self) -> UntypedLazyFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedLazyFrame(_data=self._data, _backend=self._backend)

    def with_raw(self, fn: Callable[[Any], Any]) -> LazyFrame[S]:
        """Apply a function to the raw engine LazyFrame and re-wrap.

        The function receives the underlying engine LazyFrame and must
        return the same type.  The result is wrapped back into
        ``LazyFrame[S]`` with the same schema and backend.

        Validation is deferred — it runs at ``collect()`` time if enabled,
        not at ``with_raw()`` time.
        """
        _require_backend(self._backend)
        result = fn(self._data)
        return LazyFrame(_data=result, _schema=self._schema, _backend=self._backend)

    # --- Validation ---

    def validate(self) -> LazyFrame[S]:
        """Validate that the data conforms to the schema."""
        if self._backend and self._schema:
            self._backend.validate_schema(self._data, self._schema)
        return self


# ---------------------------------------------------------------------------
# GroupBy[S] and LazyGroupBy[S]
# ---------------------------------------------------------------------------


class GroupBy(Generic[S]):
    """GroupBy on a materialized DataFrame."""

    __slots__ = ("_data", "_schema", "_keys", "_backend")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema: type[Any] | None = None,
        _keys: tuple[Column[Any], ...] = (),
        _backend: BackendProtocol | None = None,
    ) -> None:
        self._data = _data
        self._schema = _schema
        self._keys = _keys
        self._backend = _backend

    def agg(self, *exprs: AliasedExpr[Any]) -> DataFrame[Any]:
        """Aggregate grouped data. Returns DataFrame[Any] — use cast_schema()."""
        data = _require_backend(self._backend).group_by_agg(self._data, self._keys, exprs)
        return DataFrame(_data=data, _schema=None, _backend=self._backend)


class LazyGroupBy(Generic[S]):
    """GroupBy on a lazy query plan."""

    __slots__ = ("_data", "_schema", "_keys", "_backend")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema: type[Any] | None = None,
        _keys: tuple[Column[Any], ...] = (),
        _backend: BackendProtocol | None = None,
    ) -> None:
        self._data = _data
        self._schema = _schema
        self._keys = _keys
        self._backend = _backend

    def agg(self, *exprs: AliasedExpr[Any]) -> LazyFrame[Any]:
        """Aggregate grouped data. Returns LazyFrame[Any] — use cast_schema()."""
        data = _require_backend(self._backend).group_by_agg(self._data, self._keys, exprs)
        return LazyFrame(_data=data, _schema=None, _backend=self._backend)


# ---------------------------------------------------------------------------
# JoinedDataFrame[S, S2] — result of joining two DataFrames
# ---------------------------------------------------------------------------


class JoinedDataFrame(Generic[S, S2]):
    """A transitional typed DataFrame resulting from a join of two schemas.

    Operations accept columns from either schema S or S2. Available operations
    are limited to filtering, sorting, and other row-level transforms. Use
    ``cast_schema()`` to flatten into a ``DataFrame[S3]`` before group_by,
    head/tail/sample, or passing to functions that expect a single schema.
    """

    __slots__ = ("_data", "_schema_left", "_schema_right", "_backend")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema_left: type[Any] | None = None,
        _schema_right: type[Any] | None = None,
        _backend: BackendProtocol | None = None,
    ) -> None:
        self._data = _data
        self._schema_left = _schema_left
        self._schema_right = _schema_right
        self._backend = _backend

    def __repr__(self) -> str:
        left = self._schema_left.__name__ if self._schema_left else "Any"
        right = self._schema_right.__name__ if self._schema_right else "Any"
        header = f"JoinedDataFrame[{left}, {right}]"
        if self._data is not None and hasattr(self._data, "__repr__"):
            return f"{header}\n{self._data!r}"
        return header

    def _repr_html_(self) -> str | None:
        """Rich HTML representation for Jupyter notebooks."""
        left = self._schema_left.__name__ if self._schema_left else "Any"
        right = self._schema_right.__name__ if self._schema_right else "Any"
        header = f"<b>JoinedDataFrame[{left}, {right}]</b>"
        if self._data is not None and hasattr(self._data, "_repr_html_"):
            return f"{header}\n{self._data._repr_html_()}"
        return None

    def to_native(self) -> Any:
        """Return the underlying backend-native data object (e.g. pl.DataFrame)."""
        return self._data

    # --- Schema-preserving operations (return JoinedDataFrame[S, S2]) ---

    def _joined(self, data: Any) -> JoinedDataFrame[S, S2]:
        return JoinedDataFrame(
            _data=data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
            _backend=self._backend,
        )

    def filter(self, predicate: Expr[Bool]) -> JoinedDataFrame[S, S2]:
        """Filter rows by a boolean expression."""
        data = _require_backend(self._backend).filter(self._data, predicate)
        return self._joined(data)

    def sort(
        self, *columns: Column[Any] | SortExpr, descending: bool = False
    ) -> JoinedDataFrame[S, S2]:
        """Sort rows by columns or sort expressions."""
        data = _require_backend(self._backend).sort(self._data, columns, descending)
        return self._joined(data)

    def limit(self, n: int) -> JoinedDataFrame[S, S2]:
        """Limit to the first n rows."""
        data = _require_backend(self._backend).limit(self._data, n)
        return self._joined(data)

    def unique(self, *columns: Column[Any]) -> JoinedDataFrame[S, S2]:
        """Remove duplicate rows based on the given columns."""
        data = _require_backend(self._backend).unique(self._data, columns)
        return self._joined(data)

    def drop_nulls(self, *columns: Column[Any]) -> JoinedDataFrame[S, S2]:
        """Drop rows with null values in the given columns."""
        data = _require_backend(self._backend).drop_nulls(self._data, columns)
        return self._joined(data)

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> JoinedDataFrame[S, S2]:
        """Add or overwrite columns."""
        data = _require_backend(self._backend).with_columns(self._data, exprs)
        return self._joined(data)

    # --- Schema-transforming operations (return DataFrame[Any]) ---

    @overload
    def select(self, c1: Column[Any], /) -> DataFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], /) -> DataFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], c3: Column[Any], /) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        /,
    ) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        c10: Column[Any],
        /,
    ) -> DataFrame[Any]: ...

    def select(self, *columns: Column[Any]) -> DataFrame[Any]:
        """Select columns. Returns DataFrame[Any] — use cast_schema() to bind."""
        data = _require_backend(self._backend).select(self._data, columns)
        return DataFrame(_data=data, _schema=None, _backend=self._backend)

    # --- Schema transition ---

    def cast_schema(
        self,
        schema: type[S3],
        mapping: dict[Column[Any], Column[Any]] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> DataFrame[S3]:
        """Flatten join result into a single-schema DataFrame."""
        backend = _require_backend(self._backend)
        source_columns: dict[str, Column[Any]] = {}
        ambiguous: set[str] = set()
        if self._schema_left is not None:
            source_columns.update(self._schema_left._columns)
        if self._schema_right is not None:
            for name, col in self._schema_right._columns.items():
                if name in source_columns:
                    ambiguous.add(name)
                source_columns[name] = col
        name_map = _resolve_mapping(schema, source_columns, mapping, extra, ambiguous)
        data = backend.cast_schema(self._data, name_map)
        return DataFrame(_data=data, _schema=schema, _backend=self._backend)

    # --- Conversion ---

    def lazy(self) -> JoinedLazyFrame[S, S2]:
        """Convert to a lazy query plan."""
        data = _require_backend(self._backend).lazy(self._data)
        return JoinedLazyFrame(
            _data=data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
            _backend=self._backend,
        )

    def untyped(self) -> UntypedDataFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedDataFrame(_data=self._data, _backend=self._backend)


# ---------------------------------------------------------------------------
# JoinedLazyFrame[S, S2] — lazy result of joining two LazyFrames
# ---------------------------------------------------------------------------


class JoinedLazyFrame(Generic[S, S2]):
    """A transitional typed lazy query plan resulting from a join of two schemas.

    Available operations are limited to filtering, sorting, and other row-level
    transforms. Use ``cast_schema()`` to flatten into a ``LazyFrame[S3]``
    before group_by or passing to functions that expect a single schema.
    """

    __slots__ = ("_data", "_schema_left", "_schema_right", "_backend")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema_left: type[Any] | None = None,
        _schema_right: type[Any] | None = None,
        _backend: BackendProtocol | None = None,
    ) -> None:
        self._data = _data
        self._schema_left = _schema_left
        self._schema_right = _schema_right
        self._backend = _backend

    def __repr__(self) -> str:
        left = self._schema_left.__name__ if self._schema_left else "Any"
        right = self._schema_right.__name__ if self._schema_right else "Any"
        header = f"JoinedLazyFrame[{left}, {right}]"
        if self._data is not None and hasattr(self._data, "__repr__"):
            return f"{header}\n{self._data!r}"
        return header

    def _repr_html_(self) -> str | None:
        """Rich HTML representation for Jupyter notebooks."""
        left = self._schema_left.__name__ if self._schema_left else "Any"
        right = self._schema_right.__name__ if self._schema_right else "Any"
        header = f"<b>JoinedLazyFrame[{left}, {right}]</b>"
        if self._data is not None and hasattr(self._data, "_repr_html_"):
            return f"{header}\n{self._data._repr_html_()}"
        return None

    def to_native(self) -> Any:
        """Return the underlying backend-native data object (e.g. pl.LazyFrame)."""
        return self._data

    # --- Schema-preserving operations (return JoinedLazyFrame[S, S2]) ---

    def _joined(self, data: Any) -> JoinedLazyFrame[S, S2]:
        return JoinedLazyFrame(
            _data=data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
            _backend=self._backend,
        )

    def filter(self, predicate: Expr[Bool]) -> JoinedLazyFrame[S, S2]:
        """Filter rows by a boolean expression."""
        data = _require_backend(self._backend).filter(self._data, predicate)
        return self._joined(data)

    def sort(
        self, *columns: Column[Any] | SortExpr, descending: bool = False
    ) -> JoinedLazyFrame[S, S2]:
        """Sort rows by columns or sort expressions."""
        data = _require_backend(self._backend).sort(self._data, columns, descending)
        return self._joined(data)

    def limit(self, n: int) -> JoinedLazyFrame[S, S2]:
        """Limit to the first n rows."""
        data = _require_backend(self._backend).limit(self._data, n)
        return self._joined(data)

    def unique(self, *columns: Column[Any]) -> JoinedLazyFrame[S, S2]:
        """Remove duplicate rows based on the given columns."""
        data = _require_backend(self._backend).unique(self._data, columns)
        return self._joined(data)

    def drop_nulls(self, *columns: Column[Any]) -> JoinedLazyFrame[S, S2]:
        """Drop rows with null values in the given columns."""
        data = _require_backend(self._backend).drop_nulls(self._data, columns)
        return self._joined(data)

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> JoinedLazyFrame[S, S2]:
        """Add or overwrite columns."""
        data = _require_backend(self._backend).with_columns(self._data, exprs)
        return self._joined(data)

    # --- Schema-transforming operations (return LazyFrame[Any]) ---

    @overload
    def select(self, c1: Column[Any], /) -> LazyFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], /) -> LazyFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any], c2: Column[Any], c3: Column[Any], /) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any],
        c2: Column[Any],
        c3: Column[Any],
        c4: Column[Any],
        c5: Column[Any],
        c6: Column[Any],
        c7: Column[Any],
        c8: Column[Any],
        c9: Column[Any],
        c10: Column[Any],
        /,
    ) -> LazyFrame[Any]: ...

    def select(self, *columns: Column[Any]) -> LazyFrame[Any]:
        """Select columns. Returns LazyFrame[Any] — use cast_schema() to bind."""
        data = _require_backend(self._backend).select(self._data, columns)
        return LazyFrame(_data=data, _schema=None, _backend=self._backend)

    # --- Schema transition ---

    def cast_schema(
        self,
        schema: type[S3],
        mapping: dict[Column[Any], Column[Any]] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> LazyFrame[S3]:
        """Flatten join result into a single-schema LazyFrame."""
        backend = _require_backend(self._backend)
        source_columns: dict[str, Column[Any]] = {}
        ambiguous: set[str] = set()
        if self._schema_left is not None:
            source_columns.update(self._schema_left._columns)
        if self._schema_right is not None:
            for name, col in self._schema_right._columns.items():
                if name in source_columns:
                    ambiguous.add(name)
                source_columns[name] = col
        name_map = _resolve_mapping(schema, source_columns, mapping, extra, ambiguous)
        data = backend.cast_schema(self._data, name_map)
        return LazyFrame(_data=data, _schema=schema, _backend=self._backend)

    # --- Materialization ---

    def collect(self) -> JoinedDataFrame[S, S2]:
        """Materialize the lazy query plan into a JoinedDataFrame."""
        data = _require_backend(self._backend).collect(self._data)
        return JoinedDataFrame(
            _data=data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
            _backend=self._backend,
        )

    # --- Conversion ---

    def untyped(self) -> UntypedLazyFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedLazyFrame(_data=self._data, _backend=self._backend)


# ---------------------------------------------------------------------------
# Untyped escape hatches
# ---------------------------------------------------------------------------


class UntypedDataFrame:
    """A DataFrame with no schema parameter. String-based column access.

    Operations delegate to the backend when available.
    """

    __slots__ = ("_data", "_backend")

    def __init__(self, *, _data: Any = None, _backend: BackendProtocol | None = None) -> None:
        self._data = _data
        self._backend = _backend

    def select(self, *columns: str) -> UntypedDataFrame:
        """Select columns by name."""
        data = _require_backend(self._backend).select(self._data, columns)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def filter(self, expr: Any) -> UntypedDataFrame:
        """Filter rows."""
        data = _require_backend(self._backend).filter(self._data, expr)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def with_columns(self, *exprs: Any) -> UntypedDataFrame:
        """Add or overwrite columns."""
        data = _require_backend(self._backend).with_columns(self._data, exprs)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def sort(self, *columns: str, descending: bool = False) -> UntypedDataFrame:
        """Sort rows by column names."""
        data = _require_backend(self._backend).sort(self._data, columns, descending)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def limit(self, n: int) -> UntypedDataFrame:
        """Limit to the first n rows."""
        data = _require_backend(self._backend).limit(self._data, n)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def head(self, n: int = 5) -> UntypedDataFrame:
        """Return the first n rows."""
        data = _require_backend(self._backend).head(self._data, n)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def tail(self, n: int = 5) -> UntypedDataFrame:
        """Return the last n rows."""
        data = _require_backend(self._backend).tail(self._data, n)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def to_typed(self, schema: type[S]) -> DataFrame[S]:
        """Bind to a schema."""
        return DataFrame(_data=self._data, _schema=schema, _backend=self._backend)


class UntypedLazyFrame:
    """A LazyFrame with no schema parameter. String-based column access.

    Operations delegate to the backend when available.
    """

    __slots__ = ("_data", "_backend")

    def __init__(self, *, _data: Any = None, _backend: BackendProtocol | None = None) -> None:
        self._data = _data
        self._backend = _backend

    def select(self, *columns: str) -> UntypedLazyFrame:
        """Select columns by name."""
        data = _require_backend(self._backend).select(self._data, columns)
        return UntypedLazyFrame(_data=data, _backend=self._backend)

    def filter(self, expr: Any) -> UntypedLazyFrame:
        """Filter rows."""
        data = _require_backend(self._backend).filter(self._data, expr)
        return UntypedLazyFrame(_data=data, _backend=self._backend)

    def with_columns(self, *exprs: Any) -> UntypedLazyFrame:
        """Add or overwrite columns."""
        data = _require_backend(self._backend).with_columns(self._data, exprs)
        return UntypedLazyFrame(_data=data, _backend=self._backend)

    def sort(self, *columns: str, descending: bool = False) -> UntypedLazyFrame:
        """Sort rows by column names."""
        data = _require_backend(self._backend).sort(self._data, columns, descending)
        return UntypedLazyFrame(_data=data, _backend=self._backend)

    def limit(self, n: int) -> UntypedLazyFrame:
        """Limit to the first n rows."""
        data = _require_backend(self._backend).limit(self._data, n)
        return UntypedLazyFrame(_data=data, _backend=self._backend)

    def collect(self) -> UntypedDataFrame:
        """Materialize the lazy query plan."""
        data = _require_backend(self._backend).collect(self._data)
        return UntypedDataFrame(_data=data, _backend=self._backend)

    def to_typed(self, schema: type[S]) -> LazyFrame[S]:
        """Bind to a schema."""
        return LazyFrame(_data=self._data, _schema=schema, _backend=self._backend)
