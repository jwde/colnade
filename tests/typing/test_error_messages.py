"""Error message scenarios from §11.

This file is checked by ty — it must produce zero type errors.

Documents the actual error behavior for key error scenarios described in §11.
Each function demonstrates a specific error category, with comments explaining
what the type checker catches and what it doesn't.

These tests serve as documentation for developers — showing what happens
when they make common mistakes.
"""

from colnade import (
    Column,
    DataFrame,
    Float64,
    Schema,
    UInt8,
    UInt32,
    UInt64,
    Utf8,
    mapped_from,
)

# --- Schema definitions ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8 | None]
    score: Column[Float64]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[Float64]


class AgeStats(Schema):
    avg_score: Column[Float64]
    user_count: Column[UInt32]


# ===================================================================
# §11.1 Misspelled column
#
# Accessing a non-existent attribute on a Schema class produces a standard
# attribute error. Type checkers already produce good messages for this:
#   ty: Type "type[Users]" has no attribute "agee"
# ===================================================================


def check_misspelled_column() -> None:
    """§11.1: Accessing a non-existent column produces an attribute error."""
    _ = Users.agee  # type: ignore[unresolved-attribute]


# ===================================================================
# §11.2 Wrong type operation
#
# KNOWN LIMITATION: Without self-narrowing, all Column methods (sum, mean,
# str_contains, dt_year, etc.) are available on every Column regardless of
# dtype. So Users.name.sum() is NOT caught at the type level.
#
# When ty adds self-narrowing support, .sum() would only be available on
# Column[NumericType], .str_contains() only on Column[Utf8], etc.
# See AGENTS.md "Self Narrowing Limitation" for details.
# ===================================================================


def check_wrong_type_op_not_caught() -> None:
    """§11.2: Wrong-type operations are NOT caught (self-narrowing limitation).

    These calls are type-valid today but semantically wrong.
    They will fail at runtime when the backend tries to execute them.
    """
    _ = Users.name.sum()  # Should be error: sum() on Utf8
    _ = Users.name.is_nan()  # Should be error: is_nan() on Utf8
    # No type: ignore needed — these are currently accepted


# ===================================================================
# §11.3 Column from wrong schema (partial check)
#
# Expressions erase source schema: Orders.amount > 100 and Users.age > 18
# both produce Expr[Bool]. So filter/with_columns can't check schema origin.
#
# However, select/sort/group_by DO take Column[Any] directly, ensuring
# the argument is a Column instance (not a string).
# ===================================================================


def check_wrong_schema_in_filter_not_caught(df: DataFrame[Users]) -> None:
    """§11.3: Wrong-schema column in filter is NOT caught statically.

    Orders.amount > 100 produces Expr[Bool], same as Users.age > 18.
    The type checker cannot distinguish them — fails at runtime instead.
    """
    _ = df.filter(Orders.amount > 100)  # Accepted by type checker, fails at runtime


# ===================================================================
# §11.4 Schema mismatch at boundary
#
# ty error: "DataFrame[Users]" is not assignable to "DataFrame[Orders]"
# This is standard generic invariance — the type checker catches it.
# ===================================================================


def check_neg_schema_mismatch_at_boundary() -> None:
    """§11.4: Returning wrong schema type is caught."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    _: DataFrame[Orders] = df  # type: ignore[invalid-assignment]


# ===================================================================
# §11.5 Incompatible as_column binding
#
# KNOWN LIMITATION: as_column accepts Column[Any] and returns
# AliasedExpr[Any], so type mismatches between the expression type
# and the target column type are not caught.
#
# Example: Users.age.mean() returns Agg[Float64], but
#          .as_column(AgeStats.user_count) where user_count is Column[UInt32]
#          should ideally be an error. But Column[Any] erases the dtype.
#
# The spec (§11.5) envisions this check, but it would require
# as_column(target: Column[DType]) -> AliasedExpr[DType] with DType
# matching between the expression and target, which is too restrictive
# for general use (e.g., cast-then-alias patterns).
# ===================================================================


def check_incompatible_as_column_not_caught() -> None:
    """§11.5: Incompatible as_column NOT caught (Column[Any] erases dtype)."""
    _ = Users.age.mean().as_column(AgeStats.user_count)  # Should be error, not caught


# ===================================================================
# §11.6 Nullability mismatch
#
# When mapped_from maps a nullable column to a non-nullable annotation,
# the type checker catches the invariance mismatch.
#
# mapped_from(Users.age) returns Column[UInt8 | None] (preserving source type).
# Column[UInt8 | None] is NOT assignable to Column[UInt8] due to invariance.
# ===================================================================


class _NullMismatchTarget(Schema):
    """§11.6: mapped_from nullable → non-nullable is a type error."""

    age: Column[UInt8] = mapped_from(Users.age)  # type: ignore[invalid-assignment]
    # Users.age is Column[UInt8 | None], annotation expects Column[UInt8]
