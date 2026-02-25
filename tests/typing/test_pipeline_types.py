"""Cross-cutting pipeline type tests.

This file is checked by ty — it must produce zero type errors.

Tests verify that schema types propagate correctly through multi-step pipelines
spanning multiple colnade layers (schema → expression → DataFrame → LazyFrame →
GroupBy → join → cast_schema).

Unlike single-layer tests in other files, these tests combine multiple operations
to verify type preservation across layer boundaries.
"""

from colnade import (
    Column,
    DataFrame,
    Float64,
    GroupBy,
    JoinCondition,
    JoinedDataFrame,
    JoinedLazyFrame,
    LazyFrame,
    LazyGroupBy,
    Schema,
    UInt8,
    UInt64,
    Utf8,
    mapped_from,
)

# --- Schema definitions ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[Float64]


class UserSummary(Schema):
    user_name: Column[Utf8] = mapped_from(Users.name)
    user_age: Column[UInt8] = mapped_from(Users.age)


class UserSpend(Schema):
    user_id: Column[UInt64] = mapped_from(Users.id)
    user_name: Column[Utf8] = mapped_from(Users.name)
    amount: Column[Float64] = mapped_from(Orders.amount)


# --- Multi-step eager pipelines ---


def check_eager_filter_sort_limit(df: DataFrame[Users]) -> DataFrame[Users]:
    """Pipeline: filter → sort → limit preserves schema."""
    return df.filter(Users.age > 18).sort(Users.name).limit(100)


def check_eager_filter_with_columns(df: DataFrame[Users]) -> DataFrame[Users]:
    """Pipeline: filter → with_columns preserves schema."""
    return df.filter(Users.age > 18).with_columns(Users.age + 1)


def check_eager_unique_drop_nulls(df: DataFrame[Users]) -> DataFrame[Users]:
    """Pipeline: unique → drop_nulls preserves schema."""
    return df.unique(Users.id).drop_nulls(Users.name)


# --- Multi-step lazy pipelines ---


def check_lazy_full_pipeline(lf: LazyFrame[Users]) -> DataFrame[Users]:
    """Pipeline: LazyFrame → filter → sort → limit → collect → DataFrame."""
    return lf.filter(Users.age > 18).sort(Users.name).limit(100).collect()


def check_lazy_with_columns_collect(lf: LazyFrame[Users]) -> DataFrame[Users]:
    """Pipeline: lazy filter → with_columns → collect preserves schema."""
    return lf.filter(Users.age > 18).with_columns(Users.age + 1).collect()


# --- Eager → Lazy → Eager roundtrip ---


def check_eager_lazy_roundtrip(df: DataFrame[Users]) -> DataFrame[Users]:
    """Pipeline: DataFrame → lazy → filter → collect → DataFrame."""
    return df.lazy().filter(Users.age > 18).sort(Users.name).collect()


# --- GroupBy pipeline ---


def check_group_by_chain(df: DataFrame[Users]) -> GroupBy[Users]:
    """Pipeline: DataFrame → filter → group_by preserves schema."""
    return df.filter(Users.age > 18).group_by(Users.age)


def check_lazy_group_by_chain(lf: LazyFrame[Users]) -> LazyGroupBy[Users]:
    """Pipeline: LazyFrame → filter → group_by preserves schema."""
    return lf.filter(Users.age > 18).group_by(Users.age)


# --- Join pipeline ---


def check_join_pipeline(
    users: DataFrame[Users],
    orders: DataFrame[Orders],
    cond: JoinCondition,
) -> JoinedDataFrame[Users, Orders]:
    """Pipeline: filter → join preserves joined types."""
    return users.filter(Users.age > 18).join(orders, on=cond)


def check_lazy_join_pipeline(
    users: LazyFrame[Users],
    orders: LazyFrame[Orders],
    cond: JoinCondition,
) -> JoinedLazyFrame[Users, Orders]:
    """Pipeline: lazy filter → join preserves joined types."""
    return users.filter(Users.age > 18).join(orders, on=cond)


def check_join_cast_pipeline(
    j: JoinedDataFrame[Users, Orders],
) -> DataFrame[UserSpend]:
    """Pipeline: JoinedDataFrame → cast_schema exits joined world."""
    return j.cast_schema(UserSpend)


def check_lazy_join_collect_cast(
    j: JoinedLazyFrame[Users, Orders],
) -> DataFrame[UserSpend]:
    """Pipeline: JoinedLazyFrame → collect → cast_schema."""
    return j.collect().cast_schema(UserSpend)


# --- cast_schema pipeline ---


def check_filter_then_cast(df: DataFrame[Users]) -> DataFrame[UserSummary]:
    """Pipeline: filter → cast_schema changes schema."""
    return df.filter(Users.age > 18).cast_schema(UserSummary)


def check_lazy_cast_pipeline(lf: LazyFrame[Users]) -> LazyFrame[UserSummary]:
    """Pipeline: lazy filter → cast_schema changes schema at lazy level."""
    return lf.filter(Users.age > 18).cast_schema(UserSummary)


# --- Validate in pipeline ---


def check_validate_in_pipeline(df: DataFrame[Users]) -> DataFrame[Users]:
    """Pipeline: filter → validate → sort preserves schema."""
    return df.filter(Users.age > 18).validate().sort(Users.name)


# ---------------------------------------------------------------------------
# Negative type tests — regression guards
#
# Each line below MUST produce a type error, suppressed by an ignore comment.
# If types regress, the error disappears, the suppression becomes unused,
# and ty reports unused-ignore-comment — failing CI.
# ---------------------------------------------------------------------------


def check_neg_lazy_not_eager_in_pipeline() -> None:
    """Lazy pipeline result is NOT assignable to DataFrame without collect."""
    lf: LazyFrame[Users] = LazyFrame(_schema=Users)
    result = lf.filter(Users.age > 18).sort(Users.name)
    _: DataFrame[Users] = result  # type: ignore[invalid-assignment]


def check_neg_cast_changes_schema() -> None:
    """cast_schema result type is the target, NOT the source."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result = df.cast_schema(UserSummary)
    _: DataFrame[Users] = result  # type: ignore[invalid-assignment]


def check_neg_joined_must_cast() -> None:
    """JoinedDataFrame is NOT assignable to DataFrame — must cast_schema first."""
    j: JoinedDataFrame[Users, Orders] = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
    _: DataFrame[Users] = j  # type: ignore[invalid-assignment]
