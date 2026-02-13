"""DataFrame[S], LazyFrame[S], GroupBy, and untyped escape hatches.

Typed DataFrame interfaces parameterized by Schema. At this stage these are
interface definitions with stub method bodies — actual backend execution is
delegated to a backend adapter (wired in Issue #9).

Limitation: Column[DType] has a single type parameter (no schema binding),
so methods like select(), sort(), group_by() accept Column[Any] and cannot
statically enforce that columns belong to the DataFrame's schema S. Schema
enforcement at the column level requires a second type parameter on Column,
which was dropped due to Python 3.10 lacking TypeVar defaults (PEP 696).
See AGENTS.md "Column[DType] Annotation Pattern" for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, overload

from colnade.schema import S2, Column, S

if TYPE_CHECKING:
    from typing import Literal

    from colnade.dtypes import Bool
    from colnade.expr import AliasedExpr, Expr, JoinCondition, SortExpr


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

    __slots__ = ("_data", "_schema")

    def __init__(self, *, _data: Any = None, _schema: type[Any] | None = None) -> None:
        self._data = _data
        self._schema = _schema

    def __repr__(self) -> str:
        schema_name = self._schema.__name__ if self._schema else "Any"
        return f"DataFrame[{schema_name}]"

    # --- Schema-preserving operations (return DataFrame[S]) ---

    def filter(self, predicate: Expr[Bool]) -> DataFrame[S]:
        """Filter rows by a boolean expression."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def sort(self, *columns: Column[Any] | SortExpr, descending: bool = False) -> DataFrame[S]:
        """Sort rows by columns or sort expressions."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def limit(self, n: int) -> DataFrame[S]:
        """Limit to the first n rows."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def head(self, n: int = 5) -> DataFrame[S]:
        """Return the first n rows (materialized only)."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def tail(self, n: int = 5) -> DataFrame[S]:
        """Return the last n rows (materialized only)."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def sample(self, n: int) -> DataFrame[S]:
        """Return a random sample of n rows (materialized only)."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def unique(self, *columns: Column[Any]) -> DataFrame[S]:
        """Remove duplicate rows based on the given columns."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def drop_nulls(self, *columns: Column[Any]) -> DataFrame[S]:
        """Drop rows with null values in the given columns."""
        return DataFrame(_data=self._data, _schema=self._schema)

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> DataFrame[S]:
        """Add or overwrite columns. Returns DataFrame[S] (optimistic).

        Schema preservation is assumed — use ``cast_schema()`` to validate
        and bind to a new schema if columns were added or changed types.
        """
        return DataFrame(_data=self._data, _schema=self._schema)

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

    def select(self, *columns: Column[Any]) -> DataFrame[Any]:
        """Select columns. Returns DataFrame[Any] — use cast_schema() to bind."""
        return DataFrame(_data=self._data, _schema=None)

    # --- GroupBy ---

    def group_by(self, *keys: Column[Any]) -> GroupBy[S]:
        """Group by columns for aggregation."""
        return GroupBy(_data=self._data, _schema=self._schema, _keys=keys)

    # --- Join ---

    def join(
        self,
        other: DataFrame[S2],
        on: JoinCondition,
        how: Literal["inner", "left", "outer", "cross"] = "inner",
    ) -> JoinedDataFrame[S, S2]:
        """Join with another DataFrame on a JoinCondition."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema,
            _schema_right=other._schema,
        )

    # --- Conversion ---

    def lazy(self) -> LazyFrame[S]:
        """Convert to a lazy query plan."""
        return LazyFrame(_data=self._data, _schema=self._schema)

    def untyped(self) -> UntypedDataFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedDataFrame(_data=self._data)

    # --- Boundary stubs (deferred to later issues) ---

    def validate(self) -> DataFrame[S]:
        """Validate that the data conforms to the schema (Issue #8)."""
        return self

    def to_batches(self, batch_size: int | None = None) -> Any:
        """Convert to Arrow batches (Issue #9)."""
        raise NotImplementedError("to_batches requires a backend adapter (Issue #9)")


# ---------------------------------------------------------------------------
# LazyFrame[S] — typed, lazy query plan
# ---------------------------------------------------------------------------


class LazyFrame(Generic[S]):
    """A typed, lazy query plan parameterized by a Schema.

    Same operations as DataFrame except: no head(), tail(), sample(),
    to_batches() (materialized-only ops). Use collect() to materialize.
    """

    __slots__ = ("_data", "_schema")

    def __init__(self, *, _data: Any = None, _schema: type[Any] | None = None) -> None:
        self._data = _data
        self._schema = _schema

    def __repr__(self) -> str:
        schema_name = self._schema.__name__ if self._schema else "Any"
        return f"LazyFrame[{schema_name}]"

    # --- Schema-preserving operations (return LazyFrame[S]) ---

    def filter(self, predicate: Expr[Bool]) -> LazyFrame[S]:
        """Filter rows by a boolean expression."""
        return LazyFrame(_data=self._data, _schema=self._schema)

    def sort(self, *columns: Column[Any] | SortExpr, descending: bool = False) -> LazyFrame[S]:
        """Sort rows by columns or sort expressions."""
        return LazyFrame(_data=self._data, _schema=self._schema)

    def limit(self, n: int) -> LazyFrame[S]:
        """Limit to the first n rows."""
        return LazyFrame(_data=self._data, _schema=self._schema)

    def unique(self, *columns: Column[Any]) -> LazyFrame[S]:
        """Remove duplicate rows based on the given columns."""
        return LazyFrame(_data=self._data, _schema=self._schema)

    def drop_nulls(self, *columns: Column[Any]) -> LazyFrame[S]:
        """Drop rows with null values in the given columns."""
        return LazyFrame(_data=self._data, _schema=self._schema)

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> LazyFrame[S]:
        """Add or overwrite columns. Returns LazyFrame[S] (optimistic)."""
        return LazyFrame(_data=self._data, _schema=self._schema)

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

    def select(self, *columns: Column[Any]) -> LazyFrame[Any]:
        """Select columns. Returns LazyFrame[Any] — use cast_schema() to bind."""
        return LazyFrame(_data=self._data, _schema=None)

    # --- GroupBy ---

    def group_by(self, *keys: Column[Any]) -> LazyGroupBy[S]:
        """Group by columns for aggregation."""
        return LazyGroupBy(_data=self._data, _schema=self._schema, _keys=keys)

    # --- Join ---

    def join(
        self,
        other: LazyFrame[S2],
        on: JoinCondition,
        how: Literal["inner", "left", "outer", "cross"] = "inner",
    ) -> JoinedLazyFrame[S, S2]:
        """Join with another LazyFrame on a JoinCondition."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema,
            _schema_right=other._schema,
        )

    # --- Materialization ---

    def collect(self) -> DataFrame[S]:
        """Materialize the lazy query plan into a DataFrame."""
        return DataFrame(_data=self._data, _schema=self._schema)

    # --- Conversion ---

    def untyped(self) -> UntypedLazyFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedLazyFrame(_data=self._data)

    # --- Boundary stubs ---

    def validate(self) -> LazyFrame[S]:
        """Validate that the data conforms to the schema (Issue #8)."""
        return self


# ---------------------------------------------------------------------------
# GroupBy[S] and LazyGroupBy[S]
# ---------------------------------------------------------------------------


class GroupBy(Generic[S]):
    """GroupBy on a materialized DataFrame.

    Created by ``DataFrame.group_by()``. Use ``.agg()`` to produce
    aggregated results (returns ``DataFrame[Any]`` requiring ``cast_schema``).
    """

    __slots__ = ("_data", "_schema", "_keys")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema: type[Any] | None = None,
        _keys: tuple[Column[Any], ...] = (),
    ) -> None:
        self._data = _data
        self._schema = _schema
        self._keys = _keys

    def agg(self, *exprs: AliasedExpr[Any]) -> DataFrame[Any]:
        """Aggregate grouped data. Returns DataFrame[Any] — use cast_schema()."""
        return DataFrame(_data=self._data, _schema=None)


class LazyGroupBy(Generic[S]):
    """GroupBy on a lazy query plan.

    Created by ``LazyFrame.group_by()``. Use ``.agg()`` to produce
    aggregated results (returns ``LazyFrame[Any]`` requiring ``cast_schema``).
    """

    __slots__ = ("_data", "_schema", "_keys")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema: type[Any] | None = None,
        _keys: tuple[Column[Any], ...] = (),
    ) -> None:
        self._data = _data
        self._schema = _schema
        self._keys = _keys

    def agg(self, *exprs: AliasedExpr[Any]) -> LazyFrame[Any]:
        """Aggregate grouped data. Returns LazyFrame[Any] — use cast_schema()."""
        return LazyFrame(_data=self._data, _schema=None)


# ---------------------------------------------------------------------------
# JoinedDataFrame[S, S2] — result of joining two DataFrames
# ---------------------------------------------------------------------------


class JoinedDataFrame(Generic[S, S2]):
    """A typed DataFrame resulting from a join of two schemas.

    Operations accept columns from either schema S or S2. Schema-preserving
    operations return ``JoinedDataFrame[S, S2]``. Schema-transforming operations
    (select) return ``DataFrame[Any]`` and require ``cast_schema()`` to bind.

    Limitation: Column[DType] has no schema type parameter, so methods accept
    ``Column[Any]`` — cannot statically enforce columns belong to S or S2.
    """

    __slots__ = ("_data", "_schema_left", "_schema_right")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema_left: type[Any] | None = None,
        _schema_right: type[Any] | None = None,
    ) -> None:
        self._data = _data
        self._schema_left = _schema_left
        self._schema_right = _schema_right

    def __repr__(self) -> str:
        left = self._schema_left.__name__ if self._schema_left else "Any"
        right = self._schema_right.__name__ if self._schema_right else "Any"
        return f"JoinedDataFrame[{left}, {right}]"

    # --- Schema-preserving operations (return JoinedDataFrame[S, S2]) ---

    def filter(self, predicate: Expr[Bool]) -> JoinedDataFrame[S, S2]:
        """Filter rows by a boolean expression."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def sort(
        self, *columns: Column[Any] | SortExpr, descending: bool = False
    ) -> JoinedDataFrame[S, S2]:
        """Sort rows by columns or sort expressions."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def limit(self, n: int) -> JoinedDataFrame[S, S2]:
        """Limit to the first n rows."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def head(self, n: int = 5) -> JoinedDataFrame[S, S2]:
        """Return the first n rows (materialized only)."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def tail(self, n: int = 5) -> JoinedDataFrame[S, S2]:
        """Return the last n rows (materialized only)."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def sample(self, n: int) -> JoinedDataFrame[S, S2]:
        """Return a random sample of n rows (materialized only)."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def unique(self, *columns: Column[Any]) -> JoinedDataFrame[S, S2]:
        """Remove duplicate rows based on the given columns."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def drop_nulls(self, *columns: Column[Any]) -> JoinedDataFrame[S, S2]:
        """Drop rows with null values in the given columns."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> JoinedDataFrame[S, S2]:
        """Add or overwrite columns."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

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

    def select(self, *columns: Column[Any]) -> DataFrame[Any]:
        """Select columns. Returns DataFrame[Any] — use cast_schema() to bind."""
        return DataFrame(_data=self._data, _schema=None)

    # --- Conversion ---

    def lazy(self) -> JoinedLazyFrame[S, S2]:
        """Convert to a lazy query plan."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def untyped(self) -> UntypedDataFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedDataFrame(_data=self._data)


# ---------------------------------------------------------------------------
# JoinedLazyFrame[S, S2] — lazy result of joining two LazyFrames
# ---------------------------------------------------------------------------


class JoinedLazyFrame(Generic[S, S2]):
    """A typed lazy query plan resulting from a join of two schemas.

    Same operations as JoinedDataFrame except: no head(), tail(), sample()
    (materialized-only ops). Use collect() to materialize.
    """

    __slots__ = ("_data", "_schema_left", "_schema_right")

    def __init__(
        self,
        *,
        _data: Any = None,
        _schema_left: type[Any] | None = None,
        _schema_right: type[Any] | None = None,
    ) -> None:
        self._data = _data
        self._schema_left = _schema_left
        self._schema_right = _schema_right

    def __repr__(self) -> str:
        left = self._schema_left.__name__ if self._schema_left else "Any"
        right = self._schema_right.__name__ if self._schema_right else "Any"
        return f"JoinedLazyFrame[{left}, {right}]"

    # --- Schema-preserving operations (return JoinedLazyFrame[S, S2]) ---

    def filter(self, predicate: Expr[Bool]) -> JoinedLazyFrame[S, S2]:
        """Filter rows by a boolean expression."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def sort(
        self, *columns: Column[Any] | SortExpr, descending: bool = False
    ) -> JoinedLazyFrame[S, S2]:
        """Sort rows by columns or sort expressions."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def limit(self, n: int) -> JoinedLazyFrame[S, S2]:
        """Limit to the first n rows."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def unique(self, *columns: Column[Any]) -> JoinedLazyFrame[S, S2]:
        """Remove duplicate rows based on the given columns."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def drop_nulls(self, *columns: Column[Any]) -> JoinedLazyFrame[S, S2]:
        """Drop rows with null values in the given columns."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    def with_columns(self, *exprs: AliasedExpr[Any] | Expr[Any]) -> JoinedLazyFrame[S, S2]:
        """Add or overwrite columns."""
        return JoinedLazyFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

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

    def select(self, *columns: Column[Any]) -> LazyFrame[Any]:
        """Select columns. Returns LazyFrame[Any] — use cast_schema() to bind."""
        return LazyFrame(_data=self._data, _schema=None)

    # --- Materialization ---

    def collect(self) -> JoinedDataFrame[S, S2]:
        """Materialize the lazy query plan into a JoinedDataFrame."""
        return JoinedDataFrame(
            _data=self._data,
            _schema_left=self._schema_left,
            _schema_right=self._schema_right,
        )

    # --- Conversion ---

    def untyped(self) -> UntypedLazyFrame:
        """Drop type information — string-based escape hatch."""
        return UntypedLazyFrame(_data=self._data)


# ---------------------------------------------------------------------------
# Untyped escape hatches
# ---------------------------------------------------------------------------


class UntypedDataFrame:
    """A DataFrame with no schema parameter. String-based column access.

    Created by ``DataFrame.untyped()``. Use ``to_typed()`` to re-enter
    the typed world (performs runtime validation when backend is wired).
    """

    __slots__ = ("_data",)

    def __init__(self, *, _data: Any = None) -> None:
        self._data = _data

    def select(self, *columns: str) -> UntypedDataFrame:
        """Select columns by name."""
        return UntypedDataFrame(_data=self._data)

    def filter(self, expr: Any) -> UntypedDataFrame:
        """Filter rows."""
        return UntypedDataFrame(_data=self._data)

    def with_columns(self, *exprs: Any) -> UntypedDataFrame:
        """Add or overwrite columns."""
        return UntypedDataFrame(_data=self._data)

    def sort(self, *columns: str, descending: bool = False) -> UntypedDataFrame:
        """Sort rows by column names."""
        return UntypedDataFrame(_data=self._data)

    def limit(self, n: int) -> UntypedDataFrame:
        """Limit to the first n rows."""
        return UntypedDataFrame(_data=self._data)

    def head(self, n: int = 5) -> UntypedDataFrame:
        """Return the first n rows."""
        return UntypedDataFrame(_data=self._data)

    def tail(self, n: int = 5) -> UntypedDataFrame:
        """Return the last n rows."""
        return UntypedDataFrame(_data=self._data)

    def to_typed(self, schema: type[S]) -> DataFrame[S]:
        """Bind to a schema. Performs runtime validation (when backend is wired)."""
        return DataFrame(_data=self._data, _schema=schema)


class UntypedLazyFrame:
    """A LazyFrame with no schema parameter. String-based column access.

    Created by ``LazyFrame.untyped()``. Use ``to_typed()`` to re-enter
    the typed world (performs runtime validation when backend is wired).
    """

    __slots__ = ("_data",)

    def __init__(self, *, _data: Any = None) -> None:
        self._data = _data

    def select(self, *columns: str) -> UntypedLazyFrame:
        """Select columns by name."""
        return UntypedLazyFrame(_data=self._data)

    def filter(self, expr: Any) -> UntypedLazyFrame:
        """Filter rows."""
        return UntypedLazyFrame(_data=self._data)

    def with_columns(self, *exprs: Any) -> UntypedLazyFrame:
        """Add or overwrite columns."""
        return UntypedLazyFrame(_data=self._data)

    def sort(self, *columns: str, descending: bool = False) -> UntypedLazyFrame:
        """Sort rows by column names."""
        return UntypedLazyFrame(_data=self._data)

    def limit(self, n: int) -> UntypedLazyFrame:
        """Limit to the first n rows."""
        return UntypedLazyFrame(_data=self._data)

    def collect(self) -> UntypedDataFrame:
        """Materialize the lazy query plan."""
        return UntypedDataFrame(_data=self._data)

    def to_typed(self, schema: type[S]) -> LazyFrame[S]:
        """Bind to a schema. Performs runtime validation (when backend is wired)."""
        return LazyFrame(_data=self._data, _schema=schema)
