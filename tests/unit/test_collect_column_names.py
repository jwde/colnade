"""Unit tests for collect_column_names() AST walker."""

from __future__ import annotations

from colnade import (
    Column,
    ColumnRef,
    Float64,
    Literal,
    Schema,
    UInt64,
    Utf8,
)
from colnade.expr import collect_column_names


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class Orders(Schema):
    order_id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[UInt64]


class TestCollectColumnNames:
    def test_column_ref(self) -> None:
        ref = ColumnRef(column=Users.id)
        assert collect_column_names(ref) == {"id"}

    def test_bare_column(self) -> None:
        assert collect_column_names(Users.name) == {"name"}

    def test_binop(self) -> None:
        expr = Users.age > 18
        assert collect_column_names(expr) == {"age"}

    def test_binop_two_columns(self) -> None:
        expr = Users.age + Users.score
        assert collect_column_names(expr) == {"age", "score"}

    def test_unary_op(self) -> None:
        expr = Users.age.is_null()
        assert collect_column_names(expr) == {"age"}

    def test_invert(self) -> None:
        expr = ~(Users.age > 18)
        assert collect_column_names(expr) == {"age"}

    def test_literal_only(self) -> None:
        assert collect_column_names(Literal(value=42)) == set()

    def test_function_call(self) -> None:
        expr = Users.name.str_contains("Smith")
        assert collect_column_names(expr) == {"name"}

    def test_agg(self) -> None:
        expr = Users.score.sum()
        assert collect_column_names(expr) == {"score"}

    def test_aliased_expr_source_only(self) -> None:
        """AliasedExpr collects the inner expr columns but NOT the target."""
        expr = Users.score.mean().as_column(Orders.amount)
        names = collect_column_names(expr)
        assert "score" in names
        # target column is NOT collected (it's an output binding)
        assert "amount" not in names

    def test_sort_expr(self) -> None:
        expr = Users.age.desc()
        assert collect_column_names(expr) == {"age"}

    def test_logical_and(self) -> None:
        expr = (Users.age > 18) & (Users.score > 50.0)
        assert collect_column_names(expr) == {"age", "score"}

    def test_logical_or(self) -> None:
        expr = (Users.age > 18) | (Users.name == "Alice")
        assert collect_column_names(expr) == {"age", "name"}

    def test_nested_arithmetic(self) -> None:
        expr = (Users.age + 1) * Users.score
        assert collect_column_names(expr) == {"age", "score"}

    def test_multiple_args(self) -> None:
        names = collect_column_names(Users.id, Users.name, Users.age)
        assert names == {"id", "name", "age"}

    def test_mixed_args(self) -> None:
        expr = Users.age > 18
        names = collect_column_names(expr, Users.name, Users.score.desc())
        assert names == {"age", "name", "score"}

    def test_cross_schema_columns(self) -> None:
        """Columns from different schemas are all collected."""
        names = collect_column_names(Users.id, Orders.amount)
        assert names == {"id", "amount"}

    def test_fill_null(self) -> None:
        expr = Users.age.fill_null(0)
        assert collect_column_names(expr) == {"age"}

    def test_over_window(self) -> None:
        expr = Users.score.over(Users.id)
        assert collect_column_names(expr) == {"score", "id"}

    def test_empty_args(self) -> None:
        assert collect_column_names() == set()

    def test_unknown_type_ignored(self) -> None:
        assert collect_column_names(42, "hello", None) == set()
