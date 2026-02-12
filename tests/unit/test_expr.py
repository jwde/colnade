"""Unit tests for colnade.expr — AST nodes, Column expression building, lit()."""

from __future__ import annotations

from colnade import (
    Agg,
    AliasedExpr,
    BinOp,
    ColumnRef,
    Datetime,
    Float64,
    FunctionCall,
    Int32,
    Literal,
    Schema,
    SortExpr,
    UInt8,
    UInt32,
    UInt64,
    UnaryOp,
    Utf8,
    lit,
)

# ---------------------------------------------------------------------------
# Test fixture schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: UInt64
    name: Utf8
    age: UInt8 | None
    score: Float64
    created_at: Datetime


class AgeStats(Schema):
    avg_score: Float64
    user_count: UInt32


# ---------------------------------------------------------------------------
# AST node construction
# ---------------------------------------------------------------------------


class TestASTNodes:
    def test_column_ref(self) -> None:
        ref = ColumnRef(column=Users.id)
        assert ref.column is Users.id

    def test_binop(self) -> None:
        left = ColumnRef(column=Users.age)
        right = Literal(value=18)
        node = BinOp(left=left, right=right, op=">")
        assert node.left is left
        assert node.right is right
        assert node.op == ">"

    def test_unary_op(self) -> None:
        operand = ColumnRef(column=Users.age)
        node = UnaryOp(operand=operand, op="is_null")
        assert node.operand is operand
        assert node.op == "is_null"

    def test_literal(self) -> None:
        node = Literal(value=42)
        assert node.value == 42

    def test_function_call(self) -> None:
        ref = ColumnRef(column=Users.name)
        node = FunctionCall(name="str_contains", args=(ref, "Smith"))
        assert node.name == "str_contains"
        assert node.args == (ref, "Smith")
        assert node.kwargs == {}

    def test_function_call_with_kwargs(self) -> None:
        ref = ColumnRef(column=Users.score)
        node = FunctionCall(name="cast", args=(ref,), kwargs={"dtype": Int32})
        assert node.kwargs == {"dtype": Int32}

    def test_agg(self) -> None:
        ref = ColumnRef(column=Users.score)
        node = Agg(source=ref, agg_type="sum")
        assert node.source is ref
        assert node.agg_type == "sum"

    def test_aliased_expr(self) -> None:
        agg = Agg(source=ColumnRef(column=Users.score), agg_type="mean")
        node = AliasedExpr(expr=agg, target=AgeStats.avg_score)
        assert node.expr is agg
        assert node.target is AgeStats.avg_score

    def test_sort_expr(self) -> None:
        ref = ColumnRef(column=Users.age)
        node = SortExpr(expr=ref, descending=True)
        assert node.expr is ref
        assert node.descending is True


# ---------------------------------------------------------------------------
# Column comparison operators
# ---------------------------------------------------------------------------


class TestComparisonOperators:
    def test_gt(self) -> None:
        e = Users.age > 18
        assert isinstance(e, BinOp)
        assert e.op == ">"
        assert isinstance(e.left, ColumnRef)
        assert isinstance(e.right, Literal)
        assert e.right.value == 18

    def test_lt(self) -> None:
        e = Users.age < 18
        assert isinstance(e, BinOp) and e.op == "<"

    def test_ge(self) -> None:
        e = Users.age >= 18
        assert isinstance(e, BinOp) and e.op == ">="

    def test_le(self) -> None:
        e = Users.age <= 18
        assert isinstance(e, BinOp) and e.op == "<="

    def test_eq(self) -> None:
        e = Users.age == 18
        assert isinstance(e, BinOp) and e.op == "=="

    def test_ne(self) -> None:
        e = Users.age != 18
        assert isinstance(e, BinOp) and e.op == "!="

    def test_column_vs_column(self) -> None:
        e = Users.age >= Users.score
        assert isinstance(e, BinOp) and e.op == ">="
        assert isinstance(e.left, ColumnRef)
        assert isinstance(e.right, ColumnRef)
        assert e.right.column is Users.score


# ---------------------------------------------------------------------------
# Column arithmetic operators
# ---------------------------------------------------------------------------


class TestArithmeticOperators:
    def test_add(self) -> None:
        e = Users.age + 1
        assert isinstance(e, BinOp) and e.op == "+"

    def test_sub(self) -> None:
        e = Users.age - 1
        assert isinstance(e, BinOp) and e.op == "-"

    def test_mul(self) -> None:
        e = Users.age * 2
        assert isinstance(e, BinOp) and e.op == "*"

    def test_truediv(self) -> None:
        e = Users.age / 2
        assert isinstance(e, BinOp) and e.op == "/"

    def test_mod(self) -> None:
        e = Users.age % 2
        assert isinstance(e, BinOp) and e.op == "%"

    def test_column_vs_column(self) -> None:
        e = Users.age * Users.score
        assert isinstance(e, BinOp) and e.op == "*"
        assert isinstance(e.left, ColumnRef)
        assert isinstance(e.right, ColumnRef)

    def test_radd(self) -> None:
        e = 1 + Users.age
        assert isinstance(e, BinOp) and e.op == "+"
        assert isinstance(e.left, Literal) and e.left.value == 1
        assert isinstance(e.right, ColumnRef)

    def test_rsub(self) -> None:
        e = 10 - Users.age
        assert isinstance(e, BinOp) and e.op == "-"
        assert isinstance(e.left, Literal) and e.left.value == 10

    def test_rmul(self) -> None:
        e = 2 * Users.age
        assert isinstance(e, BinOp) and e.op == "*"
        assert isinstance(e.left, Literal) and e.left.value == 2

    def test_rtruediv(self) -> None:
        e = 100 / Users.age
        assert isinstance(e, BinOp) and e.op == "/"
        assert isinstance(e.left, Literal)

    def test_rmod(self) -> None:
        e = 100 % Users.age
        assert isinstance(e, BinOp) and e.op == "%"

    def test_neg(self) -> None:
        e = -Users.age
        assert isinstance(e, UnaryOp) and e.op == "-"


# ---------------------------------------------------------------------------
# Logical operators (on expressions)
# ---------------------------------------------------------------------------


class TestLogicalOperators:
    def test_and(self) -> None:
        e = (Users.age > 18) & (Users.score > 50.0)
        assert isinstance(e, BinOp) and e.op == "&"
        assert isinstance(e.left, BinOp) and e.left.op == ">"
        assert isinstance(e.right, BinOp) and e.right.op == ">"

    def test_or(self) -> None:
        e = (Users.age > 18) | (Users.score > 50.0)
        assert isinstance(e, BinOp) and e.op == "|"

    def test_invert(self) -> None:
        e = ~(Users.age > 18)
        assert isinstance(e, UnaryOp) and e.op == "~"
        assert isinstance(e.operand, BinOp)


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------


class TestAggregationMethods:
    def test_sum(self) -> None:
        e = Users.score.sum()
        assert isinstance(e, Agg) and e.agg_type == "sum"
        assert isinstance(e.source, ColumnRef) and e.source.column is Users.score

    def test_mean(self) -> None:
        e = Users.score.mean()
        assert isinstance(e, Agg) and e.agg_type == "mean"

    def test_count(self) -> None:
        e = Users.id.count()
        assert isinstance(e, Agg) and e.agg_type == "count"

    def test_min(self) -> None:
        e = Users.score.min()
        assert isinstance(e, Agg) and e.agg_type == "min"

    def test_max(self) -> None:
        e = Users.score.max()
        assert isinstance(e, Agg) and e.agg_type == "max"

    def test_std(self) -> None:
        e = Users.score.std()
        assert isinstance(e, Agg) and e.agg_type == "std"

    def test_var(self) -> None:
        e = Users.score.var()
        assert isinstance(e, Agg) and e.agg_type == "var"

    def test_first(self) -> None:
        e = Users.name.first()
        assert isinstance(e, Agg) and e.agg_type == "first"

    def test_last(self) -> None:
        e = Users.name.last()
        assert isinstance(e, Agg) and e.agg_type == "last"

    def test_n_unique(self) -> None:
        e = Users.name.n_unique()
        assert isinstance(e, Agg) and e.agg_type == "n_unique"


# ---------------------------------------------------------------------------
# String methods
# ---------------------------------------------------------------------------


class TestStringMethods:
    def test_str_contains(self) -> None:
        e = Users.name.str_contains("Smith")
        assert isinstance(e, FunctionCall) and e.name == "str_contains"
        assert e.args[1] == "Smith"

    def test_str_starts_with(self) -> None:
        e = Users.name.str_starts_with("A")
        assert isinstance(e, FunctionCall) and e.name == "str_starts_with"

    def test_str_ends_with(self) -> None:
        e = Users.name.str_ends_with("son")
        assert isinstance(e, FunctionCall) and e.name == "str_ends_with"

    def test_str_len(self) -> None:
        e = Users.name.str_len()
        assert isinstance(e, FunctionCall) and e.name == "str_len"

    def test_str_to_lowercase(self) -> None:
        e = Users.name.str_to_lowercase()
        assert isinstance(e, FunctionCall) and e.name == "str_to_lowercase"

    def test_str_to_uppercase(self) -> None:
        e = Users.name.str_to_uppercase()
        assert isinstance(e, FunctionCall) and e.name == "str_to_uppercase"

    def test_str_strip(self) -> None:
        e = Users.name.str_strip()
        assert isinstance(e, FunctionCall) and e.name == "str_strip"

    def test_str_replace(self) -> None:
        e = Users.name.str_replace("a", "b")
        assert isinstance(e, FunctionCall) and e.name == "str_replace"
        assert e.args[1] == "a" and e.args[2] == "b"


# ---------------------------------------------------------------------------
# Temporal methods
# ---------------------------------------------------------------------------


class TestTemporalMethods:
    def test_dt_year(self) -> None:
        e = Users.created_at.dt_year()
        assert isinstance(e, FunctionCall) and e.name == "dt_year"

    def test_dt_month(self) -> None:
        e = Users.created_at.dt_month()
        assert isinstance(e, FunctionCall) and e.name == "dt_month"

    def test_dt_day(self) -> None:
        e = Users.created_at.dt_day()
        assert isinstance(e, FunctionCall) and e.name == "dt_day"

    def test_dt_hour(self) -> None:
        e = Users.created_at.dt_hour()
        assert isinstance(e, FunctionCall) and e.name == "dt_hour"

    def test_dt_minute(self) -> None:
        e = Users.created_at.dt_minute()
        assert isinstance(e, FunctionCall) and e.name == "dt_minute"

    def test_dt_second(self) -> None:
        e = Users.created_at.dt_second()
        assert isinstance(e, FunctionCall) and e.name == "dt_second"

    def test_dt_truncate(self) -> None:
        e = Users.created_at.dt_truncate("1d")
        assert isinstance(e, FunctionCall) and e.name == "dt_truncate"
        assert e.args[1] == "1d"


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


class TestNullHandling:
    def test_is_null(self) -> None:
        e = Users.age.is_null()
        assert isinstance(e, UnaryOp) and e.op == "is_null"
        assert isinstance(e.operand, ColumnRef)

    def test_is_not_null(self) -> None:
        e = Users.age.is_not_null()
        assert isinstance(e, UnaryOp) and e.op == "is_not_null"

    def test_fill_null_scalar(self) -> None:
        e = Users.age.fill_null(0)
        assert isinstance(e, FunctionCall) and e.name == "fill_null"
        assert isinstance(e.args[1], Literal) and e.args[1].value == 0

    def test_fill_null_expr(self) -> None:
        e = Users.age.fill_null(lit(0))
        assert isinstance(e, FunctionCall) and e.name == "fill_null"
        assert isinstance(e.args[1], Literal)

    def test_assert_non_null(self) -> None:
        e = Users.age.assert_non_null()
        assert isinstance(e, FunctionCall) and e.name == "assert_non_null"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_is_nan(self) -> None:
        e = Users.score.is_nan()
        assert isinstance(e, UnaryOp) and e.op == "is_nan"

    def test_fill_nan(self) -> None:
        e = Users.score.fill_nan(0.0)
        assert isinstance(e, FunctionCall) and e.name == "fill_nan"
        assert isinstance(e.args[1], Literal)


# ---------------------------------------------------------------------------
# General methods
# ---------------------------------------------------------------------------


class TestGeneralMethods:
    def test_cast(self) -> None:
        e = Users.score.cast(Int32)
        assert isinstance(e, FunctionCall) and e.name == "cast"
        assert e.kwargs["dtype"] is Int32

    def test_alias(self) -> None:
        e = Users.score.alias(AgeStats.avg_score)
        assert isinstance(e, AliasedExpr)
        assert e.target is AgeStats.avg_score

    def test_as_column(self) -> None:
        e = Users.score.as_column(AgeStats.avg_score)
        assert isinstance(e, AliasedExpr)
        assert e.target is AgeStats.avg_score

    def test_agg_as_column(self) -> None:
        e = Users.score.mean().as_column(AgeStats.avg_score)
        assert isinstance(e, AliasedExpr)
        assert isinstance(e.expr, Agg)
        assert e.target is AgeStats.avg_score

    def test_desc(self) -> None:
        e = Users.age.desc()
        assert isinstance(e, SortExpr) and e.descending is True
        assert isinstance(e.expr, ColumnRef)

    def test_asc(self) -> None:
        e = Users.age.asc()
        assert isinstance(e, SortExpr) and e.descending is False

    def test_over(self) -> None:
        e = Users.score.over(Users.id)
        assert isinstance(e, FunctionCall) and e.name == "over"


# ---------------------------------------------------------------------------
# Expression chaining
# ---------------------------------------------------------------------------


class TestExpressionChaining:
    def test_nested_logical(self) -> None:
        e = (Users.age > 18) & (Users.score > 50.0)
        assert isinstance(e, BinOp) and e.op == "&"
        assert isinstance(e.left, BinOp) and e.left.op == ">"
        assert isinstance(e.right, BinOp) and e.right.op == ">"

    def test_arithmetic_then_comparison(self) -> None:
        e = (Users.age + 1) > 18
        assert isinstance(e, BinOp) and e.op == ">"
        assert isinstance(e.left, BinOp) and e.left.op == "+"

    def test_expr_arithmetic(self) -> None:
        e = (Users.age + 1) * 2
        assert isinstance(e, BinOp) and e.op == "*"
        assert isinstance(e.left, BinOp) and e.left.op == "+"

    def test_expr_desc(self) -> None:
        e = (Users.age + 1).desc()
        assert isinstance(e, SortExpr) and e.descending is True

    def test_expr_as_column(self) -> None:
        e = (Users.age + 1).as_column(AgeStats.avg_score)
        assert isinstance(e, AliasedExpr)


# ---------------------------------------------------------------------------
# lit() helper
# ---------------------------------------------------------------------------


class TestLit:
    def test_lit_int(self) -> None:
        e = lit(42)
        assert isinstance(e, Literal) and e.value == 42

    def test_lit_float(self) -> None:
        e = lit(3.14)
        assert isinstance(e, Literal) and e.value == 3.14

    def test_lit_str(self) -> None:
        e = lit("hello")
        assert isinstance(e, Literal) and e.value == "hello"

    def test_lit_bool(self) -> None:
        e = lit(True)
        assert isinstance(e, Literal) and e.value is True
