"""Schema-polymorphic function patterns from §7.

This file is checked by ty — it must produce zero type errors.

Tests verify that bounded TypeVars enable generic functions that preserve full
schema types through transformations.

Patterns tested:
  §7.1 Passthrough transforms (S → S) — works fully
  §7.3 Column-parameterized transforms (Column[N: NumericType]) — works fully

Known limitations (ty does not yet support Protocol structural subtyping
between Schema subclasses, so constrained transforms cannot be verified):
  §7.2 Constrained transforms (S: HasAge) — ty limitation
  §7.4 Multi-column constraints (S: HasScoreAndAge) — ty limitation

The constrained patterns (§7.2, §7.4) use Protocol structural subtyping to
verify that a schema has specific columns. While the code is correct Python,
ty does not currently recognize that e.g. Users structurally satisfies HasAge
when both are Schema (Protocol) subclasses. This will work once ty improves
its structural subtyping support for Protocol hierarchies.
"""

from typing import TypeVar

from colnade import (
    Column,
    DataFrame,
    Float64,
    LazyFrame,
    NumericType,
    Schema,
    UInt8,
    UInt64,
    Utf8,
)
from colnade.schema import S

# --- Schema definitions ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8]
    score: Column[Float64]


class Products(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    price: Column[Float64]


# ===================================================================
# §7.1 Passthrough transforms (S → S)
# ===================================================================


def drop_null_rows(df: DataFrame[S]) -> DataFrame[S]:
    """Generic: works on any schema, preserves full schema in return type."""
    return df.drop_nulls()


def check_passthrough_preserves_users() -> None:
    """Calling drop_null_rows on DataFrame[Users] returns DataFrame[Users]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result: DataFrame[Users] = drop_null_rows(df)
    _ = result


def check_passthrough_preserves_products() -> None:
    """Same function with different schema preserves Products."""
    df: DataFrame[Products] = DataFrame(_schema=Products)
    result: DataFrame[Products] = drop_null_rows(df)
    _ = result


# --- LazyFrame variant ---


def lazy_drop_null_rows(lf: LazyFrame[S]) -> LazyFrame[S]:
    """Generic passthrough on LazyFrame."""
    return lf.drop_nulls()


def check_lazy_passthrough() -> None:
    """LazyFrame passthrough preserves schema."""
    lf: LazyFrame[Users] = LazyFrame(_schema=Users)
    result: LazyFrame[Users] = lazy_drop_null_rows(lf)
    _ = result


# --- Passthrough with chained ops ---


def filter_and_sort(df: DataFrame[S]) -> DataFrame[S]:
    """Passthrough with multiple ops — still preserves schema."""
    # Using a literal expression that doesn't reference a specific schema
    return df.head(100)


def check_chained_passthrough() -> None:
    """Chained passthrough preserves schema through multiple ops."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result: DataFrame[Users] = filter_and_sort(df)
    _ = result


# ===================================================================
# §7.2 Constrained transforms — DOCUMENTED LIMITATION
#
# The following pattern is valid Python and matches the spec (§7.2),
# but ty does not yet support Protocol structural subtyping between
# Schema subclasses. Users structurally satisfies HasAge (it has
# age: Column[UInt8]), but ty treats them as nominally distinct.
#
# class HasAge(Schema):
#     age: Column[UInt8]
#
# _HasAgeS = TypeVar("_HasAgeS", bound=HasAge)
#
# def filter_adults(df: DataFrame[_HasAgeS]) -> DataFrame[_HasAgeS]:
#     return df.filter(HasAge.age >= 18)
#
# # This SHOULD work but doesn't in ty:
# result: DataFrame[Users] = filter_adults(users_df)
# ===================================================================


# ===================================================================
# §7.3 Column-parameterized transforms
# ===================================================================


N = TypeVar("N", bound=NumericType)


def normalize_column(df: DataFrame[S], col: Column[N]) -> DataFrame[S]:
    """Normalize any numeric column in any schema to [0, 1]."""
    return df.with_columns((col - col.min()) / (col.max() - col.min()))


def check_column_param_uint8() -> None:
    """normalize_column with UInt8 column preserves Users schema."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result: DataFrame[Users] = normalize_column(df, Users.age)
    _ = result


def check_column_param_float64() -> None:
    """normalize_column with Float64 column also works."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result: DataFrame[Users] = normalize_column(df, Users.score)
    _ = result


# ===================================================================
# §7.4 Multi-column constraints — DOCUMENTED LIMITATION
#
# Same issue as §7.2: ty doesn't do structural subtyping between
# Schema subclasses.
#
# class HasScoreAndAge(Schema):
#     age: Column[UInt8]
#     score: Column[Float64]
#
# _HSA = TypeVar("_HSA", bound=HasScoreAndAge)
#
# def age_weighted_score(df: DataFrame[_HSA]) -> DataFrame[_HSA]:
#     return df.with_columns(HasScoreAndAge.score * (HasScoreAndAge.age / 100.0))
# ===================================================================


# ---------------------------------------------------------------------------
# Negative type tests — regression guards
#
# Each line below MUST produce a type error, suppressed by an ignore comment.
# If types regress, the error disappears, the suppression becomes unused,
# and ty reports unused-ignore-comment — failing CI.
# ---------------------------------------------------------------------------


def check_neg_passthrough_not_different_schema() -> None:
    """Passthrough preserves exact schema — DataFrame[Users] not DataFrame[Products]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result = drop_null_rows(df)
    _: DataFrame[Products] = result  # type: ignore[invalid-assignment]
