"""Schema base class, Column descriptors, and schema metaclass.

Defines the foundational layer that makes column references statically verifiable:
- ``SchemaMeta`` — metaclass that creates Column descriptors from annotations
- ``Schema`` — base class for user-defined schemas (extends Protocol)
- ``Column[DType]`` — typed column descriptor with expression building
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from colnade._types import DType, T

if TYPE_CHECKING:
    from colnade.dtypes import Bool, Datetime, Float64, Int32, UInt32, Utf8
    from colnade.expr import (
        Agg,
        AliasedExpr,
        BinOp,
        ColumnRef,
        FunctionCall,
        JoinCondition,
        ListOp,
        SortExpr,
        StructFieldAccess,
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
# _MappedFrom sentinel (for cast_schema resolution)
# ---------------------------------------------------------------------------


class _MappedFrom(Generic[DType]):
    """Sentinel returned by ``mapped_from()``. Detected by SchemaMeta."""

    __slots__ = ("source",)

    def __init__(self, source: Column[DType]) -> None:
        self.source = source


def mapped_from(source: Column[DType]) -> Column[DType]:
    """Declare a column's source for ``cast_schema()`` resolution.

    Used in target schema definitions to map a column back to its source::

        class UsersClean(Schema):
            user_id: Column[UInt64] = mapped_from(Users.id)
            name: Column[Utf8]
    """
    return _MappedFrom(source)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SchemaError
# ---------------------------------------------------------------------------


class SchemaError(Exception):
    """Raised when data does not conform to the declared schema."""

    def __init__(
        self,
        *,
        missing_columns: list[str] | None = None,
        extra_columns: list[str] | None = None,
        type_mismatches: dict[str, tuple[str, str]] | None = None,
        null_violations: list[str] | None = None,
        value_violations: list[Any] | None = None,
    ) -> None:
        self.missing_columns = missing_columns or []
        self.extra_columns = extra_columns or []
        self.type_mismatches = type_mismatches or {}
        self.null_violations = null_violations or []
        self.value_violations = value_violations or []
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        parts: list[str] = []
        if self.missing_columns:
            parts.append(f"Missing columns: {', '.join(self.missing_columns)}")
        if self.extra_columns:
            parts.append(f"Extra columns: {', '.join(self.extra_columns)}")
        if self.type_mismatches:
            mismatches = [
                f"{col}: expected {exp}, got {got}"
                for col, (exp, got) in self.type_mismatches.items()
            ]
            parts.append(f"Type mismatches: {'; '.join(mismatches)}")
        if self.null_violations:
            parts.append(f"Null violations: {', '.join(self.null_violations)}")
        if self.value_violations:
            violations = []
            for v in self.value_violations:
                sample = repr(v.sample_values[:5])
                violations.append(
                    f"{v.column} [{v.constraint}]: {v.got_count} violations, sample={sample}"
                )
            parts.append(f"Value violations: {'; '.join(violations)}")
        return " | ".join(parts) if parts else "Schema validation failed"


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

    __slots__ = ("name", "dtype", "schema", "_mapped_from", "_field_info")

    def __init__(
        self,
        name: str,
        dtype: Any,
        schema: type,
        _mapped_from: Column[Any] | None = None,
        _field_info: Any | None = None,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.schema = schema
        self._mapped_from = _mapped_from
        self._field_info = _field_info

    def __repr__(self) -> str:
        return f"Column({self.name!r}, dtype={self.dtype}, schema={self.schema.__name__})"

    def __hash__(self) -> int:
        # Restore hashability after __eq__ override. Identity-based since
        # each Column descriptor is a unique instance created by SchemaMeta.
        return id(self)

    # --- Internal helpers ---

    def _ref(self) -> ColumnRef[DType]:
        """Create a ColumnRef for this column."""
        from colnade.expr import ColumnRef

        return ColumnRef(column=self)

    def _check_literal(self, value: Any, context: str = "") -> None:
        """Validate a literal value against this column's dtype (if validation enabled)."""
        from colnade.validation import check_literal_type

        check_literal_type(value, self.dtype, context or f"{self.schema.__name__}.{self.name}")

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
            self._check_literal(other)
            right = Literal(value=other)
        return BinOp(left=left, right=right, op=op)

    def _rbinop(self, other: Any, op: str) -> BinOp[Any]:
        """Create a BinOp with this column as the right operand (reverse ops)."""
        from colnade.expr import BinOp, ColumnRef, Literal

        self._check_literal(other)
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

    def __eq__(self, other: Any) -> BinOp[Bool] | JoinCondition:  # type: ignore[override]
        if isinstance(other, Column) and self.schema is not other.schema:
            from colnade.expr import JoinCondition as _JoinCondition

            return _JoinCondition(left=self, right=other)
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
            self._check_literal(value, f"{self.schema.__name__}.{self.name}.fill_null()")
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

        if not isinstance(value, Literal):
            self._check_literal(value, f"{self.schema.__name__}.{self.name}.fill_nan()")
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

    # --- Struct field access ---
    #
    # Limitation: .field() is available on ALL Column instances, not just
    # Column[Struct[S]]. Restricting it requires self narrowing:
    #
    #     def field(self: Column[Struct[S2]], col: Column[T]) -> StructFieldAccess[T]: ...
    #
    # ty does not yet support self narrowing on non-Protocol generic classes.
    # When it does, calling .field() on a non-struct column (e.g., Users.name)
    # would become a static type error. See §4.3.

    def field(self, col: Column[T]) -> StructFieldAccess[T]:
        """Access a field within a struct column.

        The ``col`` argument must be a Column descriptor from the struct's schema::

            Users.address.field(Address.city)  # StructFieldAccess[Utf8]
        """
        from colnade.expr import ColumnRef, StructFieldAccess

        return StructFieldAccess(struct_expr=ColumnRef(column=self), field=col)

    # --- List accessor ---
    #
    # Limitation: returns ListAccessor[Any] because extracting the element type
    # T from Column[List[T]] requires self narrowing:
    #
    #     @property
    #     def list(self: Column[List[T]]) -> ListAccessor[T]: ...
    #
    # ty does not yet support self narrowing on non-Protocol generic classes.
    # When it does, this property and all ListAccessor methods that return
    # ListOp[Any] can be tightened to preserve the element type. See §4.3.

    @property
    def list(self) -> ListAccessor[Any]:
        """Access list operations on a list column.

        Returns a ``ListAccessor`` that provides list-specific methods::

            Users.tags.list.len()          # ListOp node
            Users.tags.list.get(0)         # ListOp node
            Users.tags.list.contains("x")  # ListOp node
        """
        return ListAccessor(column=self)


# ---------------------------------------------------------------------------
# ListAccessor
# ---------------------------------------------------------------------------


class ListAccessor(Generic[DType]):
    """Typed accessor for list column operations.

    Provides list-specific methods (len, get, contains, sum, mean, min, max)
    that produce ``ListOp`` AST nodes for backend translation.

    Created via the ``.list`` property on Column::

        Users.tags.list.len()          # ListOp(op="len")
        Users.tags.list.get(0)         # ListOp(op="get", args=(0,))
        Users.tags.list.contains("x")  # ListOp(op="contains", args=("x",))

    **Type precision limitation:** Methods like ``get()``, ``sum()``, etc. return
    ``ListOp[Any]`` because the list element type ``T`` is not available — it is
    lost at the ``.list`` property boundary (see comment on ``Column.list``).
    With self narrowing support, these would become:

    - ``get(index) -> ListOp[T | None]``
    - ``sum() -> ListOp[T]``
    - ``contains(value: T) -> ListOp[Bool]``
    - etc.

    Methods with fixed return types (``len() -> ListOp[UInt32]``,
    ``contains() -> ListOp[Bool]``) are already precise.
    """

    __slots__ = ("_column",)

    def __init__(self, column: Column[Any]) -> None:
        self._column = column

    def __repr__(self) -> str:
        return f"ListAccessor({self._column!r})"

    def _list_op(self, op: str, *args: Any) -> ListOp[Any]:
        from colnade.expr import ColumnRef, ListOp

        return ListOp(list_expr=ColumnRef(column=self._column), op=op, args=args)

    def len(self) -> ListOp[UInt32]:
        return self._list_op("len")

    def get(self, index: int) -> ListOp[Any]:
        return self._list_op("get", index)

    def contains(self, value: Any) -> ListOp[Bool]:
        return self._list_op("contains", value)

    def sum(self) -> ListOp[Any]:
        return self._list_op("sum")

    def mean(self) -> ListOp[Any]:
        return self._list_op("mean")

    def min(self) -> ListOp[Any]:
        return self._list_op("min")

    def max(self) -> ListOp[Any]:
        return self._list_op("max")


# ---------------------------------------------------------------------------
# Row dataclass generation
# ---------------------------------------------------------------------------


def _build_row_class(schema_name: str, columns: dict[str, Column[Any]]) -> type:
    """Build a frozen dataclass representing a single row of the schema."""
    import dataclasses

    from colnade.validation import dtype_to_python_type

    fields: list[tuple[str, type]] = []
    for col_name, col in columns.items():
        py_type = dtype_to_python_type(col.dtype)
        fields.append((col_name, py_type))

    return dataclasses.make_dataclass(
        f"{schema_name}Row",
        fields,
        frozen=True,
        slots=True,
    )


# ---------------------------------------------------------------------------
# Schema metaclass
# ---------------------------------------------------------------------------


class SchemaMeta(type(Protocol)):  # type(Protocol) is typing._ProtocolMeta (CPython internal)
    """Metaclass for Schema that creates Column descriptors from annotations.

    At class creation time:
    1. Collects annotations from the class and all bases (MRO traversal).
    2. Creates Column descriptor objects for each non-private field.
    3. Stores column descriptors in ``cls._columns``.
    4. Registers the schema in the internal registry.

    Note: Inherits from ``type(Protocol)`` so that Schema subclasses are valid
    Protocol types for structural subtyping. This resolves to ``_ProtocolMeta``,
    a private CPython implementation detail. If this breaks on a future Python
    version, replace with ``type`` and drop Protocol compatibility.
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
        except (NameError, TypeError):
            # NameError: unresolvable forward references
            # TypeError: invalid annotation expressions
            # Fallback: collect raw annotations from MRO (base-first so children override)
            import warnings

            warnings.warn(
                f"Schema {name!r}: get_type_hints() failed, falling back to raw annotations. "
                "Forward references may not be resolved correctly.",
                stacklevel=2,
            )
            annotations = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, "__annotations__", {}))

        # Build Column descriptors for non-private annotations
        from colnade.constraints import FieldInfo, SchemaCheck

        columns: dict[str, Column[Any]] = {}
        for col_name, col_type in annotations.items():
            if col_name.startswith("_"):
                continue
            # Extract dtype from Column[DType] annotations
            dtype = _extract_dtype(col_type)
            # Check for mapped_from() or Field() default in namespace,
            # falling back to the parent Column descriptor for inherited columns.
            source_col: Column[Any] | None = None
            field_info: FieldInfo | None = None
            default = namespace.get(col_name)
            if isinstance(default, _MappedFrom):
                source_col = default.source
            elif isinstance(default, FieldInfo):
                field_info = default
                if default.mapped_from is not None:
                    source_col = default.mapped_from
            elif default is None:
                # Inherited column — look up parent descriptor via MRO
                for base in cls.__mro__[1:]:
                    parent_col = base.__dict__.get(col_name)
                    if isinstance(parent_col, Column):
                        source_col = parent_col._mapped_from
                        field_info = parent_col._field_info
                        break
            descriptor: Column[Any] = Column(
                name=col_name,
                dtype=dtype,
                schema=cls,
                _mapped_from=source_col,
                _field_info=field_info,
            )
            setattr(cls, col_name, descriptor)
            columns[col_name] = descriptor

        cls._columns = columns  # type: ignore[attr-defined]

        # Collect @schema_check methods (current class + inherited from bases)
        checks: list[SchemaCheck] = []
        for base in reversed(cls.__mro__[1:]):
            checks.extend(getattr(base, "_schema_checks", []))
        checks.extend(v for v in namespace.values() if isinstance(v, SchemaCheck))
        cls._schema_checks = checks  # type: ignore[attr-defined]

        # Generate Row dataclass for non-base schemas with columns
        if name != "Schema" and columns:
            cls.Row = _build_row_class(name, columns)  # type: ignore[attr-defined]

        # Register non-base schemas
        if name != "Schema":
            _schema_registry[name] = cls

        return cls

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        if hasattr(dtype, "__name__"):
            return dtype.__name__
        return str(dtype)

    def __repr__(cls) -> str:
        columns = getattr(cls, "_columns", {})
        if not columns:
            return cls.__name__
        cols = ", ".join(f"{c.name}: {SchemaMeta._dtype_name(c.dtype)}" for c in columns.values())
        return f"{cls.__name__}({cols})"

    def _repr_html_(cls) -> str:
        columns = getattr(cls, "_columns", {})
        if not columns:
            return f"<b>{cls.__name__}</b>"
        rows = "".join(
            f"<tr><td>{c.name}</td><td>{SchemaMeta._dtype_name(c.dtype)}</td></tr>"
            for c in columns.values()
        )
        return f"<b>{cls.__name__}</b><table><tr><th>Column</th><th>Type</th></tr>{rows}</table>"


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
    _schema_checks: list[Any]
    if TYPE_CHECKING:
        Row: type
