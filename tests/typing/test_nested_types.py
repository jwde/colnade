"""Static type tests for nested types — Struct[S] and List[T].

This file is checked by ty — it must produce zero type errors.

Tests cover:
- Struct column definitions with Column[Struct[S]] syntax
- .field() method preserving the field column's dtype
- List column definitions with Column[List[T]] syntax
- .list accessor returning ListAccessor
- ListAccessor methods with precise return types
- Expr chaining from StructFieldAccess and ListOp
- Negative tests as regression guards
"""

from colnade import (
    BinOp,
    Bool,
    Column,
    Expr,
    Float64,
    List,
    ListAccessor,
    ListOp,
    Schema,
    Struct,
    StructFieldAccess,
    UInt32,
    UInt64,
    Utf8,
)

# --- Schema definitions for testing ---


class Address(Schema):
    street: Column[Utf8]
    city: Column[Utf8]
    zip: Column[Utf8]


class GeoPoint(Schema):
    lat: Column[Float64]
    lng: Column[Float64]


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    address: Column[Struct[Address]]
    location: Column[Struct[GeoPoint] | None]
    tags: Column[List[Utf8]]
    scores: Column[List[Float64 | None]]
    friends: Column[List[UInt64] | None]


# --- Struct field access: return type preserves field dtype ---


def check_struct_field_access_utf8() -> None:
    """field() preserves the field column's dtype."""
    _: StructFieldAccess[Utf8] = Users.address.field(Address.city)


def check_struct_field_access_float64() -> None:
    """field() on GeoPoint field returns StructFieldAccess[Float64]."""
    _: StructFieldAccess[Float64] = Users.location.field(GeoPoint.lat)


def check_struct_field_is_expr() -> None:
    """StructFieldAccess is an Expr — covariance allows Expr[object]."""
    _: Expr[object] = Users.address.field(Address.city)


# --- Struct field access: Expr chaining works ---


def check_struct_field_comparison() -> None:
    """Comparison on StructFieldAccess returns BinOp[Bool]."""
    _: BinOp[Bool] = Users.address.field(Address.city) > "M"


def check_struct_field_arithmetic() -> None:
    """Arithmetic on StructFieldAccess preserves dtype."""
    _: BinOp[Float64] = Users.location.field(GeoPoint.lat) + 1.0


# --- List accessor: type and return types ---


def check_list_accessor_type() -> None:
    """The .list property returns a ListAccessor."""
    _: ListAccessor[object] = Users.tags.list


def check_list_len_returns_uint32() -> None:
    """ListAccessor.len() returns ListOp[UInt32]."""
    _: ListOp[UInt32] = Users.tags.list.len()


def check_list_contains_returns_bool() -> None:
    """ListAccessor.contains() returns ListOp[Bool]."""
    _: ListOp[Bool] = Users.tags.list.contains("admin")


def check_list_op_is_expr() -> None:
    """ListOp is an Expr — covariance allows Expr[object]."""
    _: Expr[object] = Users.tags.list.len()


# --- ListOp chaining ---


def check_list_len_comparison() -> None:
    """Comparison on ListOp returns BinOp[Bool]."""
    _: BinOp[Bool] = Users.tags.list.len() > 3


# ---------------------------------------------------------------------------
# Negative type tests — regression guards
#
# Each line below MUST produce a type error, suppressed by an ignore comment.
# If return types regress to Any, the error disappears, the suppression
# becomes unused, and ty reports unused-ignore-comment — failing CI.
# ---------------------------------------------------------------------------


def check_neg_struct_field_not_any() -> None:
    """field(Address.city) returns StructFieldAccess[Utf8], NOT StructFieldAccess[Float64]."""
    _: StructFieldAccess[Float64] = Users.address.field(Address.city)  # type: ignore[invalid-assignment]


def check_neg_struct_field_not_uint64() -> None:
    """field(GeoPoint.lat) returns StructFieldAccess[Float64], NOT StructFieldAccess[UInt64]."""
    _: StructFieldAccess[UInt64] = Users.location.field(GeoPoint.lat)  # type: ignore[invalid-assignment]


def check_neg_list_len_not_bool() -> None:
    """ListAccessor.len() returns ListOp[UInt32], NOT ListOp[Bool]."""
    _: ListOp[Bool] = Users.tags.list.len()  # type: ignore[invalid-assignment]


def check_neg_list_contains_not_uint32() -> None:
    """ListAccessor.contains() returns ListOp[Bool], NOT ListOp[UInt32]."""
    _: ListOp[UInt32] = Users.tags.list.contains("x")  # type: ignore[invalid-assignment]


def check_neg_struct_field_comparison_returns_bool() -> None:
    """Comparison on StructFieldAccess returns BinOp[Bool], NOT BinOp[Utf8]."""
    _: BinOp[Utf8] = Users.address.field(Address.city) > "M"  # type: ignore[invalid-assignment]
