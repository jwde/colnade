"""Schema base class, Column descriptors, and schema metaclass.

Defines the foundational layer that makes column references statically verifiable:
- ``SchemaMeta`` — metaclass that creates Column descriptors from annotations
- ``Schema`` — base class for user-defined schemas (extends Protocol)
- ``Column[DType]`` — typed column descriptor with expression building
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from colnade._types import DType

if TYPE_CHECKING:
    from colnade.dtypes import Bool, Datetime, Float64, Int32, UInt32, Utf8
    from colnade.expr import (
        Agg,
        AliasedExpr,
        BinOp,
        ColumnRef,
        FunctionCall,
        SortExpr,
        UnaryOp,
    )

# ---------------------------------------------------------------------------
# Schema-bound TypeVars (must live here because they reference Schema)
# ---------------------------------------------------------------------------

# Primary schema TypeVar — used in DataFrame[S], etc.
S = TypeVar("S", bound="Schema")

# Additional schema TypeVars for joins and multi-schema operations
S2 = TypeVar("S2", bound="Schema")
S3 = TypeVar("S3", bound="Schema")

# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

_schema_registry: dict[str, type[Schema]] = {}

# ---------------------------------------------------------------------------
# Column descriptor
# ---------------------------------------------------------------------------


class Column(Generic[DType]):
    """A typed reference to a named column in a schema.

    Used as the annotation type in schema definitions::

        class Users(Schema):
            id: Column[UInt64]
            name: Column[Utf8]
            age: Column[UInt8 | None]

    At the type level: ``Column[UInt8]`` tells the type checker this is a
    column holding ``UInt8`` data, with full access to expression-building
    methods (.sum(), .mean(), operators, etc.).

    At runtime: stores the column ``name``, ``dtype`` annotation, and owning
    ``schema`` class. All operator overloads and methods produce expression
    tree nodes (AST) for backend translation.
    """

    __slots__ = ("name", "dtype", "schema")

    def __init__(self, name: str, dtype: Any, schema: type) -> None:
        self.name = name
        self.dtype = dtype
        self.schema = schema

    def __repr__(self) -> str:
        return f"Column({self.name!r}, dtype={self.dtype}, schema={self.schema.__name__})"

    # --- Internal helpers ---

    def _ref(self) -> ColumnRef[DType]:
        """Create a ColumnRef for this column."""
        from colnade.expr import ColumnRef

        return ColumnRef(column=self)

    def _binop(self, other: Any, op: str) -> BinOp[Any]:
        """Create a BinOp with this column as the left operand."""
        from colnade.expr import BinOp, ColumnRef, Literal
        from colnade.expr import Expr as _Expr

        left = ColumnRef(column=self)
        if isinstance(other, _Expr):
            right = other
        elif isinstance(other, Column):
            right = ColumnRef(column=other)
        else:
            right = Literal(value=other)
        return BinOp(left=left, right=right, op=op)

    def _rbinop(self, other: Any, op: str) -> BinOp[Any]:
        """Create a BinOp with this column as the right operand (reverse ops)."""
        from colnade.expr import BinOp, ColumnRef, Literal

        return BinOp(left=Literal(value=other), right=ColumnRef(column=self), op=op)

    # --- Comparison operators → BinOp[Bool] ---

    def __gt__(self, other: Any) -> BinOp[Bool]:
        return self._binop(other, ">")

    def __lt__(self, other: Any) -> BinOp[Bool]:
        return self._binop(other, "<")

    def __ge__(self, other: Any) -> BinOp[Bool]:
        return self._binop(other, ">=")

    def __le__(self, other: Any) -> BinOp[Bool]:
        return self._binop(other, "<=")

    def __eq__(self, other: Any) -> BinOp[Bool]:  # type: ignore[override]
        return self._binop(other, "==")

    def __ne__(self, other: Any) -> BinOp[Bool]:  # type: ignore[override]
        return self._binop(other, "!=")

    # --- Arithmetic operators → BinOp[DType] ---

    def __add__(self, other: Any) -> BinOp[DType]:
        return self._binop(other, "+")

    def __radd__(self, other: Any) -> BinOp[DType]:
        return self._rbinop(other, "+")

    def __sub__(self, other: Any) -> BinOp[DType]:
        return self._binop(other, "-")

    def __rsub__(self, other: Any) -> BinOp[DType]:
        return self._rbinop(other, "-")

    def __mul__(self, other: Any) -> BinOp[DType]:
        return self._binop(other, "*")

    def __rmul__(self, other: Any) -> BinOp[DType]:
        return self._rbinop(other, "*")

    def __truediv__(self, other: Any) -> BinOp[DType]:
        return self._binop(other, "/")

    def __rtruediv__(self, other: Any) -> BinOp[DType]:
        return self._rbinop(other, "/")

    def __mod__(self, other: Any) -> BinOp[DType]:
        return self._binop(other, "%")

    def __rmod__(self, other: Any) -> BinOp[DType]:
        return self._rbinop(other, "%")

    def __neg__(self) -> UnaryOp[DType]:
        from colnade.expr import ColumnRef, UnaryOp

        return UnaryOp(operand=ColumnRef(column=self), op="-")

    # --- Aggregation methods ---

    def _agg(self, agg_type: str) -> Agg[Any]:
        from colnade.expr import Agg, ColumnRef

        return Agg(source=ColumnRef(column=self), agg_type=agg_type)

    def sum(self) -> Agg[DType]:
        return self._agg("sum")

    def mean(self) -> Agg[Float64]:
        return self._agg("mean")

    def min(self) -> Agg[DType]:
        return self._agg("min")

    def max(self) -> Agg[DType]:
        return self._agg("max")

    def count(self) -> Agg[UInt32]:
        return self._agg("count")

    def std(self) -> Agg[Float64]:
        return self._agg("std")

    def var(self) -> Agg[Float64]:
        return self._agg("var")

    def first(self) -> Agg[DType]:
        return self._agg("first")

    def last(self) -> Agg[DType]:
        return self._agg("last")

    def n_unique(self) -> Agg[UInt32]:
        return self._agg("n_unique")

    # --- String methods (Utf8 only at type level) ---

    def _str_fn(self, name: str, *args: Any) -> FunctionCall[Any]:
        from colnade.expr import ColumnRef, FunctionCall

        return FunctionCall(name=name, args=(ColumnRef(column=self), *args))

    def str_contains(self, pattern: str) -> FunctionCall[Bool]:
        return self._str_fn("str_contains", pattern)

    def str_starts_with(self, prefix: str) -> FunctionCall[Bool]:
        return self._str_fn("str_starts_with", prefix)

    def str_ends_with(self, suffix: str) -> FunctionCall[Bool]:
        return self._str_fn("str_ends_with", suffix)

    def str_len(self) -> FunctionCall[UInt32]:
        return self._str_fn("str_len")

    def str_to_lowercase(self) -> FunctionCall[Utf8]:
        return self._str_fn("str_to_lowercase")

    def str_to_uppercase(self) -> FunctionCall[Utf8]:
        return self._str_fn("str_to_uppercase")

    def str_strip(self) -> FunctionCall[Utf8]:
        return self._str_fn("str_strip")

    def str_replace(self, pattern: str, replacement: str) -> FunctionCall[Utf8]:
        return self._str_fn("str_replace", pattern, replacement)

    # --- Temporal methods (Datetime only at type level) ---

    def _dt_fn(self, name: str) -> FunctionCall[Any]:
        from colnade.expr import ColumnRef, FunctionCall

        return FunctionCall(name=name, args=(ColumnRef(column=self),))

    def dt_year(self) -> FunctionCall[Int32]:
        return self._dt_fn("dt_year")

    def dt_month(self) -> FunctionCall[Int32]:
        return self._dt_fn("dt_month")

    def dt_day(self) -> FunctionCall[Int32]:
        return self._dt_fn("dt_day")

    def dt_hour(self) -> FunctionCall[Int32]:
        return self._dt_fn("dt_hour")

    def dt_minute(self) -> FunctionCall[Int32]:
        return self._dt_fn("dt_minute")

    def dt_second(self) -> FunctionCall[Int32]:
        return self._dt_fn("dt_second")

    def dt_truncate(self, interval: str) -> FunctionCall[Datetime]:
        from colnade.expr import ColumnRef, FunctionCall

        return FunctionCall(name="dt_truncate", args=(ColumnRef(column=self), interval))

    # --- Null handling ---

    def is_null(self) -> UnaryOp[Bool]:
        from colnade.expr import ColumnRef, UnaryOp

        return UnaryOp(operand=ColumnRef(column=self), op="is_null")

    def is_not_null(self) -> UnaryOp[Bool]:
        from colnade.expr import ColumnRef, UnaryOp

        return UnaryOp(operand=ColumnRef(column=self), op="is_not_null")

    def fill_null(self, value: Any) -> FunctionCall[DType]:
        from colnade.expr import ColumnRef, FunctionCall
        from colnade.expr import Expr as _Expr

        if isinstance(value, _Expr):
            fill_arg = value
        else:
            from colnade.expr import Literal

            fill_arg = Literal(value=value)
        return FunctionCall(name="fill_null", args=(ColumnRef(column=self), fill_arg))

    def assert_non_null(self) -> FunctionCall[DType]:
        from colnade.expr import ColumnRef, FunctionCall

        return FunctionCall(name="assert_non_null", args=(ColumnRef(column=self),))

    # --- NaN handling (Float32/Float64 only at type level) ---

    def is_nan(self) -> UnaryOp[Bool]:
        from colnade.expr import ColumnRef, UnaryOp

        return UnaryOp(operand=ColumnRef(column=self), op="is_nan")

    def fill_nan(self, value: Any) -> FunctionCall[DType]:
        from colnade.expr import ColumnRef, FunctionCall, Literal

        fill_arg = value if isinstance(value, Literal) else Literal(value=value)
        return FunctionCall(name="fill_nan", args=(ColumnRef(column=self), fill_arg))

    # --- General ---

    def cast(self, new_dtype: type) -> FunctionCall[Any]:
        from colnade.expr import ColumnRef, FunctionCall

        return FunctionCall(
            name="cast", args=(ColumnRef(column=self),), kwargs={"dtype": new_dtype}
        )

    def alias(self, target: Column[Any]) -> AliasedExpr[Any]:
        from colnade.expr import AliasedExpr, ColumnRef

        return AliasedExpr(expr=ColumnRef(column=self), target=target)

    def as_column(self, target: Column[Any]) -> AliasedExpr[Any]:
        return self.alias(target)

    def over(self, *partition_by: Column[Any]) -> FunctionCall[DType]:
        from colnade.expr import ColumnRef, FunctionCall

        return FunctionCall(
            name="over",
            args=(ColumnRef(column=self), *[ColumnRef(column=c) for c in partition_by]),
        )

    def desc(self) -> SortExpr:
        from colnade.expr import ColumnRef, SortExpr

        return SortExpr(expr=ColumnRef(column=self), descending=True)

    def asc(self) -> SortExpr:
        from colnade.expr import ColumnRef, SortExpr

        return SortExpr(expr=ColumnRef(column=self), descending=False)


# ---------------------------------------------------------------------------
# Schema metaclass
# ---------------------------------------------------------------------------


class SchemaMeta(type(Protocol)):
    """Metaclass for Schema that creates Column descriptors from annotations.

    At class creation time:
    1. Collects annotations from the class and all bases (MRO traversal).
    2. Creates Column descriptor objects for each non-private field.
    3. Stores column descriptors in ``cls._columns``.
    4. Registers the schema in the internal registry.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> SchemaMeta:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Resolve annotations via get_type_hints() to handle PEP 563
        # (from __future__ import annotations) which stores annotations as strings.
        # get_type_hints() traverses the MRO and resolves forward references.
        try:
            annotations: dict[str, Any] = typing.get_type_hints(cls, include_extras=True)
        except Exception:
            # Fallback: collect raw annotations from MRO (base-first so children override)
            annotations = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, "__annotations__", {}))

        # Build Column descriptors for non-private annotations
        columns: dict[str, Column[Any]] = {}
        for col_name, col_type in annotations.items():
            if col_name.startswith("_"):
                continue
            # Extract dtype from Column[DType] annotations
            dtype = _extract_dtype(col_type)
            descriptor: Column[Any] = Column(name=col_name, dtype=dtype, schema=cls)
            setattr(cls, col_name, descriptor)
            columns[col_name] = descriptor

        cls._columns = columns  # type: ignore[attr-defined]

        # Register non-base schemas
        if name != "Schema":
            _schema_registry[name] = cls

        return cls


# ---------------------------------------------------------------------------
# Schema base class
# ---------------------------------------------------------------------------


def _extract_dtype(annotation: Any) -> Any:
    """Extract the data type from a column annotation.

    Supports both ``Column[DType]`` annotations (recommended) and bare dtype
    annotations (legacy). For ``Column[DType]``, extracts the inner type
    parameter. For bare types, returns the annotation unchanged.
    """
    origin = typing.get_origin(annotation)
    if origin is Column:
        args = typing.get_args(annotation)
        if args:
            return args[0]
    return annotation


# ---------------------------------------------------------------------------
# Schema base class
# ---------------------------------------------------------------------------


class Schema(Protocol, metaclass=SchemaMeta):
    """Base class for user-defined data schemas.

    Subclass this to define a typed schema::

        class Users(Schema):
            id: Column[UInt64]
            name: Column[Utf8]
            age: Column[UInt8 | None]

    The metaclass extracts the dtype from each ``Column[DType]`` annotation
    and creates Column descriptor instances, giving type checkers full
    visibility into column methods and operators.
    """

    # Populated by SchemaMeta; declared here for type checker visibility
    _columns: dict[str, Column[Any]]
