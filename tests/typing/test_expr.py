"""Static type tests for colnade.expr.

This file is checked by ty — it must produce zero type errors.

With the Column[DType] annotation pattern, type checkers see schema
attributes as Column instances with full access to expression methods.
This means Users.age is seen as Column[UInt8 | None] (not bare UInt8 | None),
so all Column methods are statically visible through schema attributes.
"""

from colnade import (
    Agg,
    AliasedExpr,
    BinOp,
    Column,
    ColumnRef,
    Datetime,
    Expr,
    Float64,
    FunctionCall,
    ListOp,
    Literal,
    Schema,
    SortExpr,
    StructFieldAccess,
    UInt8,
    UInt32,
    UInt64,
    UnaryOp,
    Utf8,
    lit,
)

# --- AST node classes are importable and usable as types ---


def check_ast_nodes_exist() -> None:
    _: type[Expr[object]] = Expr
    _: type[ColumnRef[object]] = ColumnRef
    _: type[BinOp[object]] = BinOp
    _: type[UnaryOp[object]] = UnaryOp
    _: type[Literal[object]] = Literal
    _: type[FunctionCall[object]] = FunctionCall
    _: type[Agg[object]] = Agg
    _: type[AliasedExpr[object]] = AliasedExpr
    _: type[SortExpr] = SortExpr
    _: type[StructFieldAccess[object]] = StructFieldAccess
    _: type[ListOp[object]] = ListOp


# --- Expr inheritance hierarchy ---


def check_expr_is_base(e: Expr[object]) -> None:
    """ColumnRef, BinOp, etc. are all subtypes of Expr."""
    _ = e


def check_columnref_is_expr(e: ColumnRef[object]) -> Expr[object]:
    return e


def check_binop_is_expr(e: BinOp[object]) -> Expr[object]:
    return e


def check_unaryop_is_expr(e: UnaryOp[object]) -> Expr[object]:
    return e


def check_literal_is_expr(e: Literal[object]) -> Expr[object]:
    return e


def check_agg_is_expr(e: Agg[object]) -> Expr[object]:
    return e


def check_aliased_is_expr(e: AliasedExpr[object]) -> Expr[object]:
    return e


# --- lit() produces Literal ---


def check_lit() -> None:
    e: Literal[object] = lit(42)
    _ = e
    f: Literal[object] = lit("hello")
    _ = f


# --- Column class methods exist (accessed on Column type directly) ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8 | None]
    score: Column[Float64]
    created_at: Column[Datetime]


class AgeStats(Schema):
    avg_score: Column[Float64]
    user_count: Column[UInt32]


def check_column_has_methods() -> None:
    """Verify Column class has expression-building methods defined."""
    c: Column[UInt64] = Column(name="id", dtype=UInt64, schema=Users)

    # Comparison operators
    _: Expr[object] = c > 18
    _: Expr[object] = c < 18
    _: Expr[object] = c >= 18
    _: Expr[object] = c <= 18

    # Arithmetic
    _: Expr[object] = c + 1
    _: Expr[object] = c - 1
    _: Expr[object] = c * 2

    # Aggregation
    _: Agg[object] = c.sum()
    _: Agg[object] = c.mean()
    _: Agg[object] = c.count()

    # General
    _: Expr[object] = c.cast(Float64)
    _: SortExpr = c.desc()
    _: SortExpr = c.asc()

    # Null handling
    _: Expr[object] = c.is_null()
    _: Expr[object] = c.fill_null(0)

    # String methods
    _: Expr[object] = c.str_contains("x")
    _: Expr[object] = c.str_len()

    # Temporal methods
    _: Expr[object] = c.dt_year()

    # NaN methods
    _: Expr[object] = c.is_nan()
    _: Expr[object] = c.fill_nan(0.0)


def check_expr_chaining() -> None:
    """Verify Expr supports chaining operators."""
    c: Column[UInt64] = Column(name="id", dtype=UInt64, schema=Users)
    e = c > 18

    # Logical chaining on Expr
    _: BinOp[object] = e & e
    _: BinOp[object] = e | e
    _: UnaryOp[object] = ~e

    # Arithmetic chaining on Expr
    e2 = c + 1
    _: BinOp[object] = e2 * 2
    _: BinOp[object] = e2 > 18

    # Aliasing on Expr
    target = Column[Float64](name="avg_score", dtype=Float64, schema=AgeStats)
    _a: AliasedExpr[object] = e2.as_column(target)
    _s: SortExpr = e2.desc()


def check_agg_aliasing() -> None:
    """Verify Agg supports .as_column()."""
    c: Column[Float64] = Column(name="score", dtype=Float64, schema=Users)
    target = Column[Float64](name="avg_score", dtype=Float64, schema=AgeStats)
    _: AliasedExpr[object] = c.mean().as_column(target)
