"""Typed expression tree — AST nodes for the Colnade expression DSL.

All operations on Column descriptors produce expression tree nodes rather than
immediately executing. Backend adapters translate these trees into engine-native
operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from colnade._types import DType

if TYPE_CHECKING:
    from colnade.dtypes import Bool
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

    def __and__(self, other: Expr[Any]) -> BinOp[Bool]:
        return BinOp(left=self, right=_wrap(other), op="&")

    def __or__(self, other: Expr[Any]) -> BinOp[Bool]:
        return BinOp(left=self, right=_wrap(other), op="|")

    def __invert__(self) -> UnaryOp[Bool]:
        return UnaryOp(operand=self, op="~")

    # --- Comparison operators (for expression chaining) ---

    def __gt__(self, other: Any) -> BinOp[Bool]:
        return BinOp(left=self, right=_wrap(other), op=">")

    def __lt__(self, other: Any) -> BinOp[Bool]:
        return BinOp(left=self, right=_wrap(other), op="<")

    def __ge__(self, other: Any) -> BinOp[Bool]:
        return BinOp(left=self, right=_wrap(other), op=">=")

    def __le__(self, other: Any) -> BinOp[Bool]:
        return BinOp(left=self, right=_wrap(other), op="<=")

    def __eq__(self, other: Any) -> BinOp[Bool]:  # type: ignore[override]
        return BinOp(left=self, right=_wrap(other), op="==")

    def __ne__(self, other: Any) -> BinOp[Bool]:  # type: ignore[override]
        return BinOp(left=self, right=_wrap(other), op="!=")

    # --- Arithmetic operators (for expression chaining) ---

    def __add__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=self, right=_wrap(other), op="+")

    def __radd__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=_wrap(other), right=self, op="+")

    def __sub__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=self, right=_wrap(other), op="-")

    def __rsub__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=_wrap(other), right=self, op="-")

    def __mul__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=self, right=_wrap(other), op="*")

    def __rmul__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=_wrap(other), right=self, op="*")

    def __truediv__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=self, right=_wrap(other), op="/")

    def __rtruediv__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=_wrap(other), right=self, op="/")

    def __mod__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=self, right=_wrap(other), op="%")

    def __rmod__(self, other: Any) -> BinOp[DType]:
        return BinOp(left=_wrap(other), right=self, op="%")

    def __neg__(self) -> UnaryOp[DType]:
        return UnaryOp(operand=self, op="-")

    # --- Aliasing ---

    def alias(self, target: Column[Any]) -> AliasedExpr[Any]:
        """Bind this expression to a target column."""
        return AliasedExpr(expr=self, target=target)

    def as_column(self, target: Column[Any]) -> AliasedExpr[Any]:
        """Bind this expression to a target column (alias for .alias())."""
        return AliasedExpr(expr=self, target=target)

    # --- Sorting ---

    def desc(self) -> SortExpr:
        """Sort descending."""
        return SortExpr(expr=self, descending=True)

    def asc(self) -> SortExpr:
        """Sort ascending."""
        return SortExpr(expr=self, descending=False)

    # --- Null handling ---

    def is_null(self) -> UnaryOp[Bool]:
        """Check if values are null."""
        return UnaryOp(operand=self, op="is_null")

    def is_not_null(self) -> UnaryOp[Bool]:
        """Check if values are not null."""
        return UnaryOp(operand=self, op="is_not_null")

    def fill_null(self, value: Any) -> FunctionCall[DType]:
        """Replace null values."""
        return FunctionCall(name="fill_null", args=(self, _wrap(value)))

    def assert_non_null(self) -> FunctionCall[DType]:
        """Assert values are non-null (runtime check)."""
        return FunctionCall(name="assert_non_null", args=(self,))

    # --- NaN handling ---

    def is_nan(self) -> UnaryOp[Bool]:
        """Check if values are NaN."""
        return UnaryOp(operand=self, op="is_nan")

    def fill_nan(self, value: Any) -> FunctionCall[DType]:
        """Replace NaN values."""
        fill_arg = value if isinstance(value, Literal) else Literal(value=value)
        return FunctionCall(name="fill_nan", args=(self, fill_arg))

    # --- Window ---

    def over(self, *partition_by: Column[Any]) -> FunctionCall[DType]:
        """Window function over partition columns."""
        return FunctionCall(name="over", args=(self, *[ColumnRef(column=c) for c in partition_by]))


# ---------------------------------------------------------------------------
# Concrete AST nodes
# ---------------------------------------------------------------------------


class ColumnRef(Expr[DType]):
    """Reference to a schema column."""

    __slots__ = ("column",)

    def __init__(self, column: Column[Any]) -> None:
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

    def __init__(self, expr: Expr[Any], target: Column[Any]) -> None:
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

    def __init__(self, struct_expr: Expr[Any], field: Column[Any]) -> None:
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
# Conditional expression
# ---------------------------------------------------------------------------


class WhenThenOtherwise(Expr[DType]):
    """Conditional expression: ``when(c1).then(v1).when(c2).then(v2).otherwise(default)``.

    Stores an ordered list of ``(condition, value)`` pairs and a default value.
    Backend adapters translate this to engine-native conditionals
    (e.g. ``pl.when().then().otherwise()`` for Polars, ``numpy.select`` for Pandas).
    """

    __slots__ = ("cases", "otherwise_expr")

    def __init__(
        self,
        cases: tuple[tuple[Expr[Bool], Expr[Any]], ...],
        otherwise_expr: Expr[Any],
    ) -> None:
        self.cases = cases
        self.otherwise_expr = otherwise_expr

    def when(self, condition: Expr[Bool] | Column[Bool]) -> _ChainedWhenBuilder:
        """Add another conditional branch."""
        return _ChainedWhenBuilder(prior=self, condition=_wrap(condition))

    def otherwise(self, value: Any) -> WhenThenOtherwise[DType]:
        """Set the default value for unmatched rows."""
        return WhenThenOtherwise(cases=self.cases, otherwise_expr=_wrap(value))

    def __repr__(self) -> str:
        cases_str = ", ".join(f"({c!r}, {v!r})" for c, v in self.cases)
        return f"WhenThenOtherwise([{cases_str}], otherwise={self.otherwise_expr!r})"


class _WhenBuilder:
    """Builder returned by ``when(condition)``. Call ``.then(value)`` to continue."""

    __slots__ = ("_condition",)

    def __init__(self, condition: Expr[Bool] | Column[Bool]) -> None:
        self._condition: Expr[Bool] = _wrap(condition)

    def then(self, value: Any) -> WhenThenOtherwise[Any]:
        """Provide the value for this branch."""
        return WhenThenOtherwise(
            cases=((self._condition, _wrap(value)),),
            otherwise_expr=Literal(value=None),
        )


class _ChainedWhenBuilder:
    """Builder for chained when. Call ``.then(value)`` to continue."""

    __slots__ = ("_prior", "_condition")

    def __init__(self, prior: WhenThenOtherwise[Any], condition: Expr[Bool] | Column[Bool]) -> None:
        self._prior = prior
        self._condition: Expr[Bool] = _wrap(condition)

    def then(self, value: Any) -> WhenThenOtherwise[Any]:
        """Provide the value for this branch."""
        return WhenThenOtherwise(
            cases=self._prior.cases + ((self._condition, _wrap(value)),),
            otherwise_expr=Literal(value=None),
        )


def when(condition: Expr[Bool] | Column[Bool]) -> _WhenBuilder:
    """Start a conditional expression.

    Usage::

        when(Users.age > 65).then("senior").otherwise("minor")

        when(Users.score > 90).then("A")
            .when(Users.score > 80).then("B")
            .otherwise("C")
    """
    return _WhenBuilder(condition=_wrap(condition))


# ---------------------------------------------------------------------------
# Join condition (not an Expr — a join predicate)
# ---------------------------------------------------------------------------


class JoinCondition:
    """A join predicate from comparing columns of two different schemas.

    Created by ``Column.__eq__`` when the two columns belong to different
    schemas::

        Users.id == Orders.user_id  # → JoinCondition
        Users.age == Users.score    # → BinOp[Bool] (same schema)
    """

    __slots__ = ("left", "right")

    def __init__(self, left: Column[Any], right: Column[Any]) -> None:
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"JoinCondition({self.left.name!r} == {self.right.name!r})"


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


# ---------------------------------------------------------------------------
# Expression tree walking
# ---------------------------------------------------------------------------


def collect_column_names(*args: Any) -> set[str]:
    """Collect all column names referenced in expressions.

    Walks expression trees, ``Column`` instances, and ``SortExpr`` wrappers.
    Returns a set of column name strings found across all arguments.
    """
    from colnade.schema import Column

    names: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, ColumnRef):
            names.add(node.column.name)
        elif isinstance(node, BinOp):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, UnaryOp):
            _walk(node.operand)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                _walk(arg)
        elif isinstance(node, Agg):
            _walk(node.source)
        elif isinstance(node, (AliasedExpr, SortExpr)):
            _walk(node.expr)
        elif isinstance(node, StructFieldAccess):
            _walk(node.struct_expr)
        elif isinstance(node, ListOp):
            _walk(node.list_expr)
        elif isinstance(node, WhenThenOtherwise):
            for cond, val in node.cases:
                _walk(cond)
                _walk(val)
            _walk(node.otherwise_expr)
        elif isinstance(node, Column):
            names.add(node.name)

    for arg in args:
        _walk(arg)

    return names
