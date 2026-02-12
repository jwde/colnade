"""Typed expression tree — AST nodes for the Colnade expression DSL.

All operations on Column descriptors produce expression tree nodes rather than
immediately executing. Backend adapters translate these trees into engine-native
operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from colnade._types import DType

if TYPE_CHECKING:
    from colnade.schema import Column


# ---------------------------------------------------------------------------
# Base expression
# ---------------------------------------------------------------------------


class Expr(Generic[DType]):
    """Base class for all expression tree nodes.

    Supports logical chaining (``&``, ``|``, ``~``), comparison,
    arithmetic operators, and aliasing.
    """

    # --- Logical operators (for chaining boolean expressions) ---

    def __and__(self, other: Expr[Any]) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="&")

    def __or__(self, other: Expr[Any]) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="|")

    def __invert__(self) -> UnaryOp[Any]:
        return UnaryOp(operand=self, op="~")

    # --- Comparison operators (for expression chaining) ---

    def __gt__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op=">")

    def __lt__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="<")

    def __ge__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op=">=")

    def __le__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="<=")

    def __eq__(self, other: Any) -> BinOp[Any]:  # type: ignore[override]
        return BinOp(left=self, right=_wrap(other), op="==")

    def __ne__(self, other: Any) -> BinOp[Any]:  # type: ignore[override]
        return BinOp(left=self, right=_wrap(other), op="!=")

    # --- Arithmetic operators (for expression chaining) ---

    def __add__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="+")

    def __radd__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=_wrap(other), right=self, op="+")

    def __sub__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="-")

    def __rsub__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=_wrap(other), right=self, op="-")

    def __mul__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="*")

    def __rmul__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=_wrap(other), right=self, op="*")

    def __truediv__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="/")

    def __rtruediv__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=_wrap(other), right=self, op="/")

    def __mod__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=self, right=_wrap(other), op="%")

    def __rmod__(self, other: Any) -> BinOp[Any]:
        return BinOp(left=_wrap(other), right=self, op="%")

    def __neg__(self) -> UnaryOp[Any]:
        return UnaryOp(operand=self, op="-")

    # --- Aliasing ---

    def alias(self, target: Column[Any, Any]) -> AliasedExpr[Any]:
        """Bind this expression to a target column."""
        return AliasedExpr(expr=self, target=target)

    def as_column(self, target: Column[Any, Any]) -> AliasedExpr[Any]:
        """Bind this expression to a target column (alias for .alias())."""
        return AliasedExpr(expr=self, target=target)

    # --- Sorting ---

    def desc(self) -> SortExpr:
        """Sort descending."""
        return SortExpr(expr=self, descending=True)

    def asc(self) -> SortExpr:
        """Sort ascending."""
        return SortExpr(expr=self, descending=False)

    # --- Window ---

    def over(self, *partition_by: Column[Any, Any]) -> Expr[Any]:
        """Window function over partition columns."""
        return FunctionCall(name="over", args=(self, *[ColumnRef(column=c) for c in partition_by]))


# ---------------------------------------------------------------------------
# Concrete AST nodes
# ---------------------------------------------------------------------------


class ColumnRef(Expr[DType]):
    """Reference to a schema column."""

    __slots__ = ("column",)

    def __init__(self, column: Column[Any, Any]) -> None:
        self.column = column

    def __repr__(self) -> str:
        return f"ColumnRef({self.column.name!r})"


class BinOp(Expr[DType]):
    """Binary operation (arithmetic, comparison, logical)."""

    __slots__ = ("left", "right", "op")

    def __init__(self, left: Expr[Any], right: Expr[Any], op: str) -> None:
        self.left = left
        self.right = right
        self.op = op

    def __repr__(self) -> str:
        return f"BinOp({self.left!r} {self.op} {self.right!r})"


class UnaryOp(Expr[DType]):
    """Unary operation (negation, not, is_null, etc.)."""

    __slots__ = ("operand", "op")

    def __init__(self, operand: Expr[Any], op: str) -> None:
        self.operand = operand
        self.op = op

    def __repr__(self) -> str:
        return f"UnaryOp({self.op} {self.operand!r})"


class Literal(Expr[DType]):
    """A literal value."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


class FunctionCall(Expr[DType]):
    """Named function application (str_contains, dt_year, cast, etc.)."""

    __slots__ = ("name", "args", "kwargs")

    def __init__(
        self,
        name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}

    def __repr__(self) -> str:
        return f"FunctionCall({self.name!r}, args={self.args!r})"


class Agg(Expr[DType]):
    """Aggregation expression (sum, mean, count, etc.)."""

    __slots__ = ("source", "agg_type")

    def __init__(self, source: Expr[Any], agg_type: str) -> None:
        self.source = source
        self.agg_type = agg_type

    def __repr__(self) -> str:
        return f"Agg({self.agg_type!r}, {self.source!r})"


class AliasedExpr(Expr[DType]):
    """Expression with an output column binding."""

    __slots__ = ("expr", "target")

    def __init__(self, expr: Expr[Any], target: Column[Any, Any]) -> None:
        self.expr = expr
        self.target = target

    def __repr__(self) -> str:
        return f"AliasedExpr({self.expr!r} -> {self.target.name!r})"


class SortExpr:
    """Sort direction wrapper. Not an expression — used in .sort() calls."""

    __slots__ = ("expr", "descending")

    def __init__(self, expr: Expr[Any], descending: bool) -> None:
        self.expr = expr
        self.descending = descending

    def __repr__(self) -> str:
        direction = "desc" if self.descending else "asc"
        return f"SortExpr({self.expr!r}, {direction})"


# ---------------------------------------------------------------------------
# Nested type AST nodes (implementations wired in issue #5)
# ---------------------------------------------------------------------------


class StructFieldAccess(Expr[DType]):
    """Access a field within a struct column."""

    __slots__ = ("struct_expr", "field")

    def __init__(self, struct_expr: Expr[Any], field: Column[Any, Any]) -> None:
        self.struct_expr = struct_expr
        self.field = field

    def __repr__(self) -> str:
        return f"StructFieldAccess({self.struct_expr!r}.{self.field.name!r})"


class ListOp(Expr[DType]):
    """Operation on a list column (len, get, contains, sum, etc.)."""

    __slots__ = ("list_expr", "op", "args")

    def __init__(self, list_expr: Expr[Any], op: str, args: tuple[Any, ...] = ()) -> None:
        self.list_expr = list_expr
        self.op = op
        self.args = args

    def __repr__(self) -> str:
        return f"ListOp({self.list_expr!r}.{self.op}({', '.join(repr(a) for a in self.args)}))"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _wrap(value: Any) -> Expr[Any]:
    """Wrap a raw value as an Expr node (ColumnRef or Literal)."""
    if isinstance(value, Expr):
        return value
    # Import Column at runtime to avoid circular import
    from colnade.schema import Column

    if isinstance(value, Column):
        return ColumnRef(column=value)
    return Literal(value=value)


def lit(value: Any) -> Literal[Any]:
    """Create a literal expression."""
    return Literal(value=value)
