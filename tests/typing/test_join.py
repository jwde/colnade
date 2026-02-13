"""Static type tests for the join system.

This file is checked by ty — it must produce zero type errors.

Tests cover:
- JoinCondition creation from cross-schema ==
- .join() return types on DataFrame and LazyFrame
- JoinedDataFrame schema-preserving ops
- JoinedLazyFrame ops and collect()
- Conversions (lazy, untyped)
- Negative regression guards
"""

from typing import Any

from colnade import (
    Column,
    DataFrame,
    JoinCondition,
    JoinedDataFrame,
    JoinedLazyFrame,
    LazyFrame,
    Schema,
    UInt64,
    UntypedDataFrame,
    UntypedLazyFrame,
    Utf8,
)

# --- Schema definitions ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[UInt64]


# --- Join return types ---


def check_dataframe_join(
    df: DataFrame[Users], orders: DataFrame[Orders], cond: JoinCondition
) -> JoinedDataFrame[Users, Orders]:
    return df.join(orders, on=cond)


def check_lazyframe_join(
    lf: LazyFrame[Users], orders: LazyFrame[Orders], cond: JoinCondition
) -> JoinedLazyFrame[Users, Orders]:
    return lf.join(orders, on=cond)


# --- JoinedDataFrame schema-preserving ops ---


def check_joined_filter(
    j: JoinedDataFrame[Users, Orders],
) -> JoinedDataFrame[Users, Orders]:
    return j.filter(Users.age > 18)


def check_joined_sort(
    j: JoinedDataFrame[Users, Orders],
) -> JoinedDataFrame[Users, Orders]:
    return j.sort(Users.name)


def check_joined_limit(
    j: JoinedDataFrame[Users, Orders],
) -> JoinedDataFrame[Users, Orders]:
    return j.limit(10)


def check_joined_unique(
    j: JoinedDataFrame[Users, Orders],
) -> JoinedDataFrame[Users, Orders]:
    return j.unique(Users.id)


def check_joined_drop_nulls(
    j: JoinedDataFrame[Users, Orders],
) -> JoinedDataFrame[Users, Orders]:
    return j.drop_nulls(Users.name)


def check_joined_with_columns(
    j: JoinedDataFrame[Users, Orders],
) -> JoinedDataFrame[Users, Orders]:
    return j.with_columns(Users.age + 1)


# --- JoinedDataFrame select returns DataFrame[Any] ---


def check_joined_select(j: JoinedDataFrame[Users, Orders]) -> None:
    _: DataFrame[Any] = j.select(Users.name, Orders.amount)


# --- JoinedDataFrame conversions ---


def check_joined_lazy(
    j: JoinedDataFrame[Users, Orders],
) -> JoinedLazyFrame[Users, Orders]:
    return j.lazy()


def check_joined_untyped(j: JoinedDataFrame[Users, Orders]) -> UntypedDataFrame:
    return j.untyped()


# --- JoinedLazyFrame ops ---


def check_joined_lazy_filter(
    j: JoinedLazyFrame[Users, Orders],
) -> JoinedLazyFrame[Users, Orders]:
    return j.filter(Users.age > 18)


def check_joined_lazy_collect(
    j: JoinedLazyFrame[Users, Orders],
) -> JoinedDataFrame[Users, Orders]:
    return j.collect()


def check_joined_lazy_untyped(
    j: JoinedLazyFrame[Users, Orders],
) -> UntypedLazyFrame:
    return j.untyped()


# ---------------------------------------------------------------------------
# Negative type tests — regression guards
# ---------------------------------------------------------------------------


def check_neg_joined_not_dataframe() -> None:
    """JoinedDataFrame[Users, Orders] is NOT assignable to DataFrame[Users]."""
    j: JoinedDataFrame[Users, Orders] = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
    _: DataFrame[Users] = j  # type: ignore[invalid-assignment]


def check_neg_joined_lazy_not_joined() -> None:
    """JoinedLazyFrame is NOT assignable to JoinedDataFrame."""
    jl: JoinedLazyFrame[Users, Orders] = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders)
    _: JoinedDataFrame[Users, Orders] = jl  # type: ignore[invalid-assignment]


def check_neg_joined_order_matters() -> None:
    """JoinedDataFrame[Users, Orders] is NOT assignable to JoinedDataFrame[Orders, Users]."""
    j: JoinedDataFrame[Users, Orders] = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
    _: JoinedDataFrame[Orders, Users] = j  # type: ignore[invalid-assignment]


def check_neg_joined_lazy_order_matters() -> None:
    """JoinedLazyFrame[Users, Orders] is NOT assignable to JoinedLazyFrame[Orders, Users]."""
    jl: JoinedLazyFrame[Users, Orders] = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders)
    _: JoinedLazyFrame[Orders, Users] = jl  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Schema-preservation guards — detect if return types regress to Any
# ---------------------------------------------------------------------------


def check_neg_joined_filter_preserves_exact_type() -> None:
    """JoinedDataFrame.filter() returns exact joined type, NOT JoinedDataFrame[Any, Any]."""
    j: JoinedDataFrame[Users, Orders] = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
    result = j.filter(Users.age > 18)
    _: JoinedDataFrame[Orders, Users] = result  # type: ignore[invalid-assignment]


def check_neg_joined_lazy_collect_preserves_type() -> None:
    """JoinedLazyFrame.collect() returns exact joined type."""
    jl: JoinedLazyFrame[Users, Orders] = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders)
    result = jl.collect()
    _: JoinedDataFrame[Orders, Users] = result  # type: ignore[invalid-assignment]
