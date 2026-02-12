"""Unit tests for nested types — Struct[S] and List[T]."""

from __future__ import annotations

from colnade import (
    Column,
    ColumnRef,
    Float64,
    List,
    ListAccessor,
    ListOp,
    Schema,
    Struct,
    StructFieldAccess,
    UInt64,
    Utf8,
)

# ---------------------------------------------------------------------------
# Test fixture schemas
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Struct in schemas
# ---------------------------------------------------------------------------


class TestStructInSchema:
    def test_struct_column_is_column(self) -> None:
        assert isinstance(Users.address, Column)

    def test_struct_column_dtype(self) -> None:
        """Users.address dtype should be Struct[Address]."""
        dtype = Users._columns["address"].dtype
        assert dtype is not None
        # The dtype should be Struct[Address] — get_origin is Struct
        import typing

        origin = typing.get_origin(dtype)
        assert origin is Struct
        args = typing.get_args(dtype)
        assert args == (Address,)

    def test_nullable_struct_column_dtype(self) -> None:
        """Users.location dtype should reflect Struct[GeoPoint] | None."""
        import types
        import typing

        dtype = Users._columns["location"].dtype
        # Should be a Union of Struct[GeoPoint] and None
        origin = typing.get_origin(dtype)
        assert origin is types.UnionType or origin is typing.Union
        args = typing.get_args(dtype)
        # One arg is Struct[GeoPoint], the other is NoneType
        type_names = {str(a) for a in args}
        assert any("Struct" in name for name in type_names)
        assert type(None) in args

    def test_struct_schema_columns_present(self) -> None:
        """Address schema should have its own columns."""
        assert "street" in Address._columns
        assert "city" in Address._columns
        assert "zip" in Address._columns


# ---------------------------------------------------------------------------
# Struct field access AST
# ---------------------------------------------------------------------------


class TestStructFieldAccess:
    def test_field_returns_struct_field_access(self) -> None:
        result = Users.address.field(Address.city)
        assert isinstance(result, StructFieldAccess)

    def test_field_struct_expr(self) -> None:
        result = Users.address.field(Address.city)
        assert isinstance(result.struct_expr, ColumnRef)
        assert result.struct_expr.column is Users.address

    def test_field_references_target_column(self) -> None:
        result = Users.address.field(Address.city)
        assert result.field is Address.city

    def test_field_different_fields(self) -> None:
        city_access = Users.address.field(Address.city)
        zip_access = Users.address.field(Address.zip)
        assert city_access.field is Address.city
        assert zip_access.field is Address.zip

    def test_field_chained_with_expr_operators(self) -> None:
        """Chained: Users.address.field(Address.city) > "M" produces BinOp."""
        from colnade import BinOp

        result = Users.address.field(Address.city) > "M"
        assert isinstance(result, BinOp)
        assert result.op == ">"
        # The left operand should be the StructFieldAccess
        assert isinstance(result.left, StructFieldAccess)

    def test_field_repr(self) -> None:
        result = Users.address.field(Address.city)
        assert "StructFieldAccess" in repr(result)
        assert "city" in repr(result)

    def test_nullable_struct_field_access(self) -> None:
        """Accessing a field through a nullable struct still produces StructFieldAccess."""
        result = Users.location.field(GeoPoint.lat)
        assert isinstance(result, StructFieldAccess)
        assert result.field is GeoPoint.lat


# ---------------------------------------------------------------------------
# List in schemas
# ---------------------------------------------------------------------------


class TestListInSchema:
    def test_list_column_is_column(self) -> None:
        assert isinstance(Users.tags, Column)

    def test_list_column_dtype(self) -> None:
        """Users.tags dtype should be List[Utf8]."""
        import typing

        dtype = Users._columns["tags"].dtype
        origin = typing.get_origin(dtype)
        assert origin is List
        args = typing.get_args(dtype)
        assert args == (Utf8,)

    def test_list_nullable_element_dtype(self) -> None:
        """Users.scores dtype should be List[Float64 | None]."""
        import types
        import typing

        dtype = Users._columns["scores"].dtype
        origin = typing.get_origin(dtype)
        assert origin is List
        args = typing.get_args(dtype)
        assert len(args) == 1
        # The element type should be Float64 | None (a union)
        elem = args[0]
        elem_origin = typing.get_origin(elem)
        assert elem_origin is types.UnionType or elem_origin is typing.Union
        elem_args = typing.get_args(elem)
        assert Float64 in elem_args
        assert type(None) in elem_args

    def test_nullable_list_dtype(self) -> None:
        """Users.friends dtype should be List[UInt64] | None."""
        import types
        import typing

        dtype = Users._columns["friends"].dtype
        origin = typing.get_origin(dtype)
        assert origin is types.UnionType or origin is typing.Union
        args = typing.get_args(dtype)
        assert type(None) in args
        # One of the args should be List[UInt64]
        list_arg = [a for a in args if typing.get_origin(a) is List]
        assert len(list_arg) == 1


# ---------------------------------------------------------------------------
# List accessor
# ---------------------------------------------------------------------------


class TestListAccessor:
    def test_list_property_returns_accessor(self) -> None:
        accessor = Users.tags.list
        assert isinstance(accessor, ListAccessor)

    def test_list_accessor_repr(self) -> None:
        accessor = Users.tags.list
        assert "ListAccessor" in repr(accessor)

    def test_list_len(self) -> None:
        result = Users.tags.list.len()
        assert isinstance(result, ListOp)
        assert result.op == "len"
        assert result.args == ()

    def test_list_get(self) -> None:
        result = Users.tags.list.get(0)
        assert isinstance(result, ListOp)
        assert result.op == "get"
        assert result.args == (0,)

    def test_list_get_negative_index(self) -> None:
        result = Users.tags.list.get(-1)
        assert isinstance(result, ListOp)
        assert result.op == "get"
        assert result.args == (-1,)

    def test_list_contains(self) -> None:
        result = Users.tags.list.contains("admin")
        assert isinstance(result, ListOp)
        assert result.op == "contains"
        assert result.args == ("admin",)

    def test_list_sum(self) -> None:
        result = Users.scores.list.sum()
        assert isinstance(result, ListOp)
        assert result.op == "sum"
        assert result.args == ()

    def test_list_mean(self) -> None:
        result = Users.scores.list.mean()
        assert isinstance(result, ListOp)
        assert result.op == "mean"
        assert result.args == ()

    def test_list_min(self) -> None:
        result = Users.scores.list.min()
        assert isinstance(result, ListOp)
        assert result.op == "min"

    def test_list_max(self) -> None:
        result = Users.scores.list.max()
        assert isinstance(result, ListOp)
        assert result.op == "max"

    def test_list_op_has_correct_list_expr(self) -> None:
        result = Users.tags.list.len()
        assert isinstance(result.list_expr, ColumnRef)
        assert result.list_expr.column is Users.tags

    def test_list_op_repr(self) -> None:
        result = Users.tags.list.len()
        assert "ListOp" in repr(result)
        assert "len" in repr(result)


# ---------------------------------------------------------------------------
# Nested nullability (runtime behavior)
# ---------------------------------------------------------------------------


class TestNestedNullability:
    def test_nullable_struct_field_produces_valid_ast(self) -> None:
        """Accessing a field through a nullable struct should work at runtime."""
        result = Users.location.field(GeoPoint.lat)
        assert isinstance(result, StructFieldAccess)
        # The struct_expr references the nullable struct column
        assert isinstance(result.struct_expr, ColumnRef)
        assert result.struct_expr.column is Users.location

    def test_nullable_list_accessor_works(self) -> None:
        """Accessing .list on a nullable list column should work at runtime."""
        result = Users.friends.list.len()
        assert isinstance(result, ListOp)
        assert result.op == "len"
