"""Static type tests for colnade.expr.

This file is checked by ty — it must produce zero type errors.

With the Column[DType] annotation pattern, type checkers see schema
attributes as Column instances with full access to expression methods.
This means Users.age is seen as Column[UInt8 | None] (not bare UInt8 | None),
so all Column methods are statically visible through schema attributes.

Return types are precise — comparisons yield BinOp[Bool], arithmetic
preserves DType, aggregations return Agg[DType|Float64|UInt32], etc.
"""

from colnade import (
    Agg,
    AliasedExpr,
    BinOp,
    Bool,
    Column,
    ColumnRef,
    Datetime,
    Expr,
    Float64,
    FunctionCall,
    Int32,
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
    WhenThenOtherwise,
    lit,
    when,
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
    _: type[WhenThenOtherwise[object]] = WhenThenOtherwise


# --- Expr inheritance hierarchy ---


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


# --- Schema definitions for testing ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8 | None]
    score: Column[Float64]
    created_at: Column[Datetime]


class AgeStats(Schema):
    avg_score: Column[Float64]
    user_count: Column[UInt32]


# --- Column methods: precise return types ---


def check_comparison_operators() -> None:
    """Comparison operators return BinOp[Bool]."""
    _: BinOp[Bool] = Users.id > 18
    _: BinOp[Bool] = Users.id < 18
    _: BinOp[Bool] = Users.id >= 18
    _: BinOp[Bool] = Users.id <= 18

    # Covariance: BinOp[Bool] is assignable to Expr[object]
    _e: Expr[object] = Users.id > 18


def check_arithmetic_operators() -> None:
    """Arithmetic operators preserve DType."""
    _: BinOp[UInt64] = Users.id + 1
    _: BinOp[UInt64] = Users.id - 1
    _: BinOp[UInt64] = Users.id * 2

    # Nullable column preserves nullable type
    _n: BinOp[UInt8 | None] = Users.age + 1


def check_negation() -> None:
    """Negation preserves DType."""
    _: UnaryOp[UInt64] = -Users.id


def check_aggregations() -> None:
    """Aggregations: sum/min/max preserve DType, mean→Float64, count→UInt32."""
    _: Agg[UInt64] = Users.id.sum()
    _: Agg[Float64] = Users.score.mean()
    _: Agg[UInt32] = Users.id.count()
    _: Agg[UInt64] = Users.id.min()
    _: Agg[UInt64] = Users.id.max()
    _: Agg[Float64] = Users.score.std()
    _: Agg[Float64] = Users.score.var()
    _: Agg[Utf8] = Users.name.first()
    _: Agg[Utf8] = Users.name.last()
    _: Agg[UInt32] = Users.name.n_unique()


def check_string_methods() -> None:
    """String methods return precise types."""
    _: FunctionCall[Bool] = Users.name.str_contains("x")
    _: FunctionCall[Bool] = Users.name.str_starts_with("A")
    _: FunctionCall[Bool] = Users.name.str_ends_with("z")
    _: FunctionCall[UInt32] = Users.name.str_len()
    _: FunctionCall[Utf8] = Users.name.str_to_lowercase()
    _: FunctionCall[Utf8] = Users.name.str_to_uppercase()
    _: FunctionCall[Utf8] = Users.name.str_strip()
    _: FunctionCall[Utf8] = Users.name.str_replace("a", "b")


def check_temporal_methods() -> None:
    """Temporal methods return FunctionCall[Int32] or FunctionCall[Datetime]."""
    _: FunctionCall[Int32] = Users.created_at.dt_year()
    _: FunctionCall[Int32] = Users.created_at.dt_month()
    _: FunctionCall[Int32] = Users.created_at.dt_day()
    _: FunctionCall[Int32] = Users.created_at.dt_hour()
    _: FunctionCall[Int32] = Users.created_at.dt_minute()
    _: FunctionCall[Int32] = Users.created_at.dt_second()
    _: FunctionCall[Datetime] = Users.created_at.dt_truncate("1d")


def check_null_handling() -> None:
    """Null methods: is_null→Bool, fill_null preserves DType."""
    _: UnaryOp[Bool] = Users.age.is_null()
    _: UnaryOp[Bool] = Users.age.is_not_null()
    _: FunctionCall[UInt8 | None] = Users.age.fill_null(0)
    _: FunctionCall[UInt8 | None] = Users.age.assert_non_null()


def check_nan_handling() -> None:
    """NaN methods: is_nan→Bool, fill_nan preserves DType."""
    _: UnaryOp[Bool] = Users.score.is_nan()
    _: FunctionCall[Float64] = Users.score.fill_nan(0.0)


def check_general_methods() -> None:
    """Cast, alias, sort, over."""
    _: FunctionCall[object] = Users.score.cast(Int32)
    _: SortExpr = Users.id.desc()
    _: SortExpr = Users.id.asc()
    _: FunctionCall[UInt64] = Users.id.over(Users.name)


def check_aliasing() -> None:
    """alias/as_column produce AliasedExpr."""
    target = Column[Float64](name="avg_score", dtype=Float64, schema=AgeStats)
    _: AliasedExpr[object] = Users.score.alias(target)
    _: AliasedExpr[object] = Users.score.as_column(target)
    _: AliasedExpr[object] = Users.score.mean().as_column(target)


# --- Expr chaining: types propagate through chains ---


def check_expr_chaining() -> None:
    """Verify Expr chaining preserves types."""
    e = Users.id > 18  # BinOp[Bool]

    # Logical chaining preserves DType (Bool here)
    _: BinOp[Bool] = e & e
    _: BinOp[Bool] = e | e
    _: UnaryOp[Bool] = ~e

    # Arithmetic chaining preserves DType
    e2 = Users.id + 1  # BinOp[UInt64]
    _: BinOp[UInt64] = e2 * 2
    _: BinOp[Bool] = e2 > 18  # comparison always → Bool

    # Aliasing on Expr
    target = Column[Float64](name="avg_score", dtype=Float64, schema=AgeStats)
    _a: AliasedExpr[object] = e2.as_column(target)
    _s: SortExpr = e2.desc()


# ---------------------------------------------------------------------------
# Negative type tests — regression guards
#
# Each line below MUST produce a type error, suppressed by an ignore comment.
# If return types regress to Any, the error disappears, the suppression
# becomes unused, and ty reports unused-ignore-comment — failing CI.
# ---------------------------------------------------------------------------


def check_neg_comparison_not_dtype() -> None:
    """Comparison returns BinOp[Bool], NOT BinOp of the column's dtype."""
    _: BinOp[UInt64] = Users.id > 1  # type: ignore[invalid-assignment]


def check_neg_arithmetic_not_bool() -> None:
    """Arithmetic preserves DType, does NOT produce Bool."""
    _: BinOp[Bool] = Users.id + 1  # type: ignore[invalid-assignment]


def check_neg_mean_returns_float64() -> None:
    """mean() returns Agg[Float64], NOT Agg of the column's dtype."""
    _: Agg[UInt64] = Users.id.mean()  # type: ignore[invalid-assignment]


def check_neg_count_returns_uint32() -> None:
    """count() returns Agg[UInt32], NOT Agg of the column's dtype."""
    _: Agg[UInt64] = Users.id.count()  # type: ignore[invalid-assignment]


def check_neg_sum_preserves_dtype() -> None:
    """sum() preserves DType — Agg[UInt64] not assignable to Agg[Float64]."""
    _: Agg[Float64] = Users.id.sum()  # type: ignore[invalid-assignment]


def check_neg_is_null_returns_bool() -> None:
    """is_null() returns UnaryOp[Bool], NOT UnaryOp of the column's dtype."""
    _: UnaryOp[UInt64] = Users.id.is_null()  # type: ignore[invalid-assignment]


def check_neg_str_contains_returns_bool() -> None:
    """str_contains returns FunctionCall[Bool], NOT FunctionCall[Utf8]."""
    _: FunctionCall[Utf8] = Users.name.str_contains("x")  # type: ignore[invalid-assignment]


def check_neg_str_len_returns_uint32() -> None:
    """str_len returns FunctionCall[UInt32], NOT FunctionCall[Bool]."""
    _: FunctionCall[Bool] = Users.name.str_len()  # type: ignore[invalid-assignment]


def check_neg_dt_year_returns_int32() -> None:
    """dt_year returns FunctionCall[Int32], NOT FunctionCall[Datetime]."""
    _: FunctionCall[Datetime] = Users.created_at.dt_year()  # type: ignore[invalid-assignment]


def check_neg_is_nan_returns_bool() -> None:
    """is_nan() returns UnaryOp[Bool], NOT UnaryOp[Float64]."""
    _: UnaryOp[Float64] = Users.score.is_nan()  # type: ignore[invalid-assignment]


def check_neg_expr_comparison_returns_bool() -> None:
    """Expr chaining: comparison on Expr also returns BinOp[Bool]."""
    e = Users.id + 1  # BinOp[UInt64]
    _: BinOp[UInt64] = e > 18  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Additional return type guards — every Column method's return type is pinned
# ---------------------------------------------------------------------------


def check_neg_std_returns_float64() -> None:
    """std() returns Agg[Float64], NOT Agg of the column's dtype."""
    _: Agg[UInt64] = Users.id.std()  # type: ignore[invalid-assignment]


def check_neg_var_returns_float64() -> None:
    """var() returns Agg[Float64], NOT Agg of the column's dtype."""
    _: Agg[UInt64] = Users.id.var()  # type: ignore[invalid-assignment]


def check_neg_first_preserves_dtype() -> None:
    """first() preserves DType — Agg[Utf8] not assignable to Agg[Float64]."""
    _: Agg[Float64] = Users.name.first()  # type: ignore[invalid-assignment]


def check_neg_last_preserves_dtype() -> None:
    """last() preserves DType — Agg[Utf8] not assignable to Agg[Float64]."""
    _: Agg[Float64] = Users.name.last()  # type: ignore[invalid-assignment]


def check_neg_n_unique_returns_uint32() -> None:
    """n_unique() returns Agg[UInt32], NOT Agg of the column's dtype."""
    _: Agg[UInt64] = Users.id.n_unique()  # type: ignore[invalid-assignment]


def check_neg_str_to_lowercase_returns_utf8() -> None:
    """str_to_lowercase returns FunctionCall[Utf8], NOT FunctionCall[Bool]."""
    _: FunctionCall[Bool] = Users.name.str_to_lowercase()  # type: ignore[invalid-assignment]


def check_neg_dt_truncate_returns_datetime() -> None:
    """dt_truncate returns FunctionCall[Datetime], NOT FunctionCall[Int32]."""
    _: FunctionCall[Int32] = Users.created_at.dt_truncate("1d")  # type: ignore[invalid-assignment]


def check_neg_fill_null_preserves_dtype() -> None:
    """fill_null preserves DType — not assignable to wrong type."""
    _: FunctionCall[Bool] = Users.age.fill_null(0)  # type: ignore[invalid-assignment]


def check_neg_fill_nan_preserves_dtype() -> None:
    """fill_nan preserves DType — not assignable to wrong type."""
    _: FunctionCall[Bool] = Users.score.fill_nan(0.0)  # type: ignore[invalid-assignment]


def check_neg_over_preserves_dtype() -> None:
    """over() preserves DType — not assignable to wrong type."""
    _: FunctionCall[Bool] = Users.id.over(Users.name)  # type: ignore[invalid-assignment]


def check_neg_negation_preserves_dtype() -> None:
    """Negation preserves DType — UnaryOp[UInt64] not assignable to UnaryOp[Bool]."""
    _: UnaryOp[Bool] = -Users.id  # type: ignore[invalid-assignment]


def check_neg_sort_expr_not_expr() -> None:
    """SortExpr is NOT an Expr — it's a separate type for sort specifications."""
    s: SortExpr = Users.id.desc()
    _: Expr[object] = s  # type: ignore[invalid-assignment]


def check_neg_logical_and_preserves_dtype() -> None:
    """Logical & preserves DType — BinOp[Bool] & BinOp[Bool] stays Bool."""
    e = Users.id > 18  # BinOp[Bool]
    _: BinOp[UInt64] = e & e  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# WhenThenOtherwise type tests
# ---------------------------------------------------------------------------


def check_when_returns_when_then_otherwise() -> None:
    """when().then().otherwise() returns WhenThenOtherwise."""
    _: WhenThenOtherwise[object] = when(Users.age > 65).then("senior").otherwise("minor")


def check_when_is_expr() -> None:
    """WhenThenOtherwise is an Expr."""
    e: WhenThenOtherwise[object] = when(Users.age > 65).then("senior").otherwise("minor")
    _: Expr[object] = e


def check_when_supports_alias() -> None:
    """alias() works on WhenThenOtherwise result."""
    _: AliasedExpr[object] = (
        when(Users.age > 65).then("senior").otherwise("minor").alias(Users.name)
    )
