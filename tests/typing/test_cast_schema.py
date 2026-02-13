"""Static type tests for cast_schema, mapped_from, and SchemaError.

This file is checked by ty — it must produce zero type errors.
"""

from typing import Any

from colnade import (
    Column,
    DataFrame,
    JoinedDataFrame,
    JoinedLazyFrame,
    LazyFrame,
    Schema,
    UInt64,
    Utf8,
    mapped_from,
)

# --- Schema definitions ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[UInt64]


class UsersSummary(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


class RenamedUsers(Schema):
    user_id: Column[UInt64] = mapped_from(Users.id)
    user_name: Column[Utf8] = mapped_from(Users.name)


# --- Positive: DataFrame.cast_schema ---


def check_df_cast_schema(df: DataFrame[Users]) -> DataFrame[UsersSummary]:
    return df.cast_schema(UsersSummary)


def check_df_cast_schema_renamed(df: DataFrame[Users]) -> DataFrame[RenamedUsers]:
    return df.cast_schema(RenamedUsers)


# --- Positive: LazyFrame.cast_schema ---


def check_lf_cast_schema(lf: LazyFrame[Users]) -> LazyFrame[UsersSummary]:
    return lf.cast_schema(UsersSummary)


# --- Positive: JoinedDataFrame.cast_schema → DataFrame ---


def check_joined_cast_schema(
    j: JoinedDataFrame[Users, Orders],
) -> DataFrame[UsersSummary]:
    return j.cast_schema(UsersSummary)


# --- Positive: JoinedLazyFrame.cast_schema → LazyFrame ---


def check_joined_lazy_cast_schema(
    j: JoinedLazyFrame[Users, Orders],
) -> LazyFrame[UsersSummary]:
    return j.cast_schema(UsersSummary)


# --- Positive: mapped_from preserves Column type ---


def check_mapped_from_type() -> Column[UInt64]:
    return mapped_from(Users.id)


def check_mapped_from_utf8() -> Column[Utf8]:
    return mapped_from(Users.name)


# ---------------------------------------------------------------------------
# Negative type tests — regression guards
# ---------------------------------------------------------------------------


def check_neg_joined_cast_not_joined() -> None:
    """JoinedDataFrame.cast_schema returns DataFrame, NOT JoinedDataFrame."""
    j: JoinedDataFrame[Users, Orders] = JoinedDataFrame(_schema_left=Users, _schema_right=Orders)
    result = j.cast_schema(UsersSummary)
    _: JoinedDataFrame[Any, Any] = result  # type: ignore[invalid-assignment]


def check_neg_joined_lazy_cast_not_joined() -> None:
    """JoinedLazyFrame.cast_schema returns LazyFrame, NOT JoinedLazyFrame."""
    j: JoinedLazyFrame[Users, Orders] = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders)
    result = j.cast_schema(UsersSummary)
    _: JoinedLazyFrame[Any, Any] = result  # type: ignore[invalid-assignment]
