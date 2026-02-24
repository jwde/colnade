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
    Date,
    Datetime,
    Float64,
    List,
    Schema,
    Struct,
    UInt8,
    UInt32,
    UInt64,
    Utf8,
    mapped_from,
)

# --- Schema definitions ---


class Address(Schema):
    city: Column[Utf8]
    zip_code: Column[Utf8]


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8 | None]
    score: Column[Float64]
    created: Column[Datetime]
    birthday: Column[Date | None]
    address: Column[Struct[Address]]
    tags: Column[List[Utf8]]


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
# §11.2 Wrong type operation — CAUGHT via self-narrowing
#
# Column methods are restricted to appropriate dtypes using self-narrowing:
#   - sum/mean/std/var → numeric types only
#   - is_nan/fill_nan → float types only
#   - str_* → Utf8 only
#   - dt_year/month/day → Date or Datetime only
#   - dt_hour/minute/second → Datetime or Time only
#   - dt_truncate → Datetime only
#   - .field() → Struct columns only
#   - .list → List columns only
# ===================================================================


# --- Positive tests (must NOT produce errors) ---


def check_numeric_agg_on_numeric() -> None:
    """§11.2+: Numeric aggregations work on numeric columns."""
    _ = Users.score.sum()  # Float64 — numeric
    _ = Users.id.mean()  # UInt64 — numeric
    _ = Users.age.sum()  # UInt8 | None — nullable numeric
    _ = Users.score.std()  # Float64 — numeric
    _ = Users.score.var()  # Float64 — numeric


def check_float_on_float() -> None:
    """§11.2+: NaN methods work on float columns."""
    _ = Users.score.is_nan()  # Float64
    _ = Users.score.fill_nan(0.0)  # Float64


def check_str_on_str() -> None:
    """§11.2+: String methods work on Utf8 columns."""
    _ = Users.name.str_contains("x")  # Utf8
    _ = Users.name.str_len()
    _ = Users.name.str_to_lowercase()
    _ = Users.name.str_starts_with("A")
    _ = Users.name.str_ends_with("z")
    _ = Users.name.str_to_uppercase()
    _ = Users.name.str_strip()
    _ = Users.name.str_replace("a", "b")


def check_temporal_on_temporal() -> None:
    """§11.2+: Temporal methods work on appropriate temporal columns."""
    _ = Users.created.dt_year()  # Datetime
    _ = Users.created.dt_hour()  # Datetime
    _ = Users.created.dt_truncate("1h")  # Datetime
    _ = Users.birthday.dt_year()  # Date | None
    _ = Users.birthday.dt_month()  # Date | None
    _ = Users.birthday.dt_day()  # Date | None


def check_struct_on_struct() -> None:
    """§11.2+: .field() works on Struct columns."""
    _ = Users.address.field(Address.city)


def check_list_on_list() -> None:
    """§11.2+: .list works on List columns."""
    _ = Users.tags.list.len()
    _ = Users.tags.list.get(0)
    _ = Users.tags.list.contains("x")


# --- Negative tests (each MUST produce an error) ---


def check_neg_numeric_agg_on_string() -> None:
    """§11.2: sum/mean/std/var on Utf8 is caught."""
    _ = Users.name.sum()  # type: ignore[invalid-argument-type]
    _ = Users.name.mean()  # type: ignore[invalid-argument-type]
    _ = Users.name.std()  # type: ignore[invalid-argument-type]
    _ = Users.name.var()  # type: ignore[invalid-argument-type]


def check_neg_float_on_non_float() -> None:
    """§11.2: is_nan/fill_nan on non-float is caught."""
    _ = Users.name.is_nan()  # type: ignore[invalid-argument-type]
    _ = Users.age.is_nan()  # type: ignore[invalid-argument-type]
    _ = Users.id.fill_nan(0)  # type: ignore[invalid-argument-type]


def check_neg_str_on_non_str() -> None:
    """§11.2: String methods on non-Utf8 is caught."""
    _ = Users.score.str_contains("x")  # type: ignore[invalid-argument-type]
    _ = Users.id.str_len()  # type: ignore[invalid-argument-type]


def check_neg_temporal_on_non_temporal() -> None:
    """§11.2: Temporal methods on non-temporal is caught."""
    _ = Users.name.dt_year()  # type: ignore[invalid-argument-type]
    _ = Users.score.dt_hour()  # type: ignore[invalid-argument-type]


def check_neg_struct_on_non_struct() -> None:
    """§11.2: .field() on non-Struct is caught."""
    _ = Users.name.field(Address.city)  # type: ignore[invalid-argument-type]


def check_neg_list_on_non_list() -> None:
    """§11.2: .list on non-List — NOT caught (property self-narrowing unsupported)."""
    _ = Users.name.list  # Accepted — ty doesn't narrow property self types


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
