"""Unit tests for colnade.schema — Schema, Column, SchemaMeta."""

from __future__ import annotations

import dataclasses
import datetime
import types
import warnings
from unittest.mock import patch

import pytest

from colnade import (
    Binary,
    Column,
    Date,
    Datetime,
    Float64,
    Schema,
    UInt8,
    UInt64,
    Utf8,
)
from colnade.schema import _schema_registry

# ---------------------------------------------------------------------------
# Test fixture schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8 | None]
    score: Column[Float64]


class EnrichedUsers(Users):
    normalized_age: Column[Float64]


class UserSummary(Schema):
    name: Column[Utf8]
    score: Column[Float64]


class HasUserId(Schema):
    user_id: Column[UInt64]


class HasTimestamp(Schema):
    created_at: Column[Datetime]


class Events(HasUserId, HasTimestamp):
    event_type: Column[Utf8]


class Empty(Schema):
    pass


class OnlyPrivate(Schema):
    _internal: int


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    def test_columns_dict_exists(self) -> None:
        assert hasattr(Users, "_columns")
        assert isinstance(Users._columns, dict)

    def test_columns_keys(self) -> None:
        assert set(Users._columns.keys()) == {"id", "name", "age", "score"}

    def test_column_is_column_instance(self) -> None:
        assert isinstance(Users.id, Column)
        assert isinstance(Users.name, Column)

    def test_column_name(self) -> None:
        assert Users.id.name == "id"
        assert Users.name.name == "name"
        assert Users.age.name == "age"
        assert Users.score.name == "score"

    def test_column_dtype(self) -> None:
        assert Users.id.dtype is UInt64
        assert Users.name.dtype is Utf8
        assert Users.score.dtype is Float64

    def test_column_schema(self) -> None:
        assert Users.id.schema is Users
        assert Users.name.schema is Users


# ---------------------------------------------------------------------------
# Column descriptor properties
# ---------------------------------------------------------------------------


class TestColumnDescriptor:
    def test_column_repr(self) -> None:
        r = repr(Users.id)
        assert "id" in r
        assert "UInt64" in r
        assert "Users" in r

    def test_column_is_generic(self) -> None:
        # Column should be subscriptable (Generic[DType])
        alias = Column[UInt64]
        assert alias is not None

    def test_private_annotations_skipped(self) -> None:
        assert "_internal" not in OnlyPrivate._columns
        assert len(OnlyPrivate._columns) == 0

    def test_column_ref(self) -> None:
        from colnade.expr import ColumnRef

        ref = Users.id._ref()
        assert isinstance(ref, ColumnRef)
        assert ref.column is Users.id


# ---------------------------------------------------------------------------
# Nullable columns
# ---------------------------------------------------------------------------


class TestNullableColumns:
    def test_nullable_dtype(self) -> None:
        # age: Column[UInt8 | None] — dtype should be the union type
        age_dtype = Users.age.dtype
        assert isinstance(age_dtype, types.UnionType)

    def test_non_nullable_dtype(self) -> None:
        # name: Column[Utf8] — dtype should be the class directly
        assert Users.name.dtype is Utf8


# ---------------------------------------------------------------------------
# Schema inheritance
# ---------------------------------------------------------------------------


class TestSchemaInheritance:
    def test_child_has_parent_columns(self) -> None:
        # EnrichedUsers extends Users
        assert "id" in EnrichedUsers._columns
        assert "name" in EnrichedUsers._columns
        assert "age" in EnrichedUsers._columns
        assert "score" in EnrichedUsers._columns

    def test_child_has_own_columns(self) -> None:
        assert "normalized_age" in EnrichedUsers._columns

    def test_child_column_count(self) -> None:
        # Users has 4, EnrichedUsers adds 1
        assert len(EnrichedUsers._columns) == 5

    def test_child_columns_rebound_to_child(self) -> None:
        # Inherited columns should be re-bound to the child schema
        assert EnrichedUsers.id.schema is EnrichedUsers
        assert EnrichedUsers.name.schema is EnrichedUsers

    def test_child_own_column_schema(self) -> None:
        assert EnrichedUsers.normalized_age.schema is EnrichedUsers


# ---------------------------------------------------------------------------
# Multiple inheritance (trait composition)
# ---------------------------------------------------------------------------


class TestTraitComposition:
    def test_events_has_all_columns(self) -> None:
        assert "user_id" in Events._columns
        assert "created_at" in Events._columns
        assert "event_type" in Events._columns

    def test_events_column_count(self) -> None:
        assert len(Events._columns) == 3

    def test_trait_columns_bound_to_composed_class(self) -> None:
        assert Events.user_id.schema is Events
        assert Events.created_at.schema is Events
        assert Events.event_type.schema is Events

    def test_trait_column_dtypes(self) -> None:
        assert Events.user_id.dtype is UInt64
        assert Events.created_at.dtype is Datetime
        assert Events.event_type.dtype is Utf8


# ---------------------------------------------------------------------------
# Empty and edge case schemas
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_schema(self) -> None:
        assert len(Empty._columns) == 0

    def test_only_private_schema(self) -> None:
        assert len(OnlyPrivate._columns) == 0


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------


class TestSchemaRegistry:
    def test_schemas_registered(self) -> None:
        assert "Users" in _schema_registry
        assert "EnrichedUsers" in _schema_registry
        assert "Events" in _schema_registry

    def test_schema_base_not_registered(self) -> None:
        assert "Schema" not in _schema_registry

    def test_registry_values_are_classes(self) -> None:
        assert _schema_registry["Users"] is Users


# ---------------------------------------------------------------------------
# SchemaMeta robustness
# ---------------------------------------------------------------------------


class TestSchemaMetaRobustness:
    def test_get_type_hints_primary_path_works(self) -> None:
        """Normal schema creation uses get_type_hints() successfully."""

        class Normal(Schema):
            x: Column[UInt64]

        assert "x" in Normal._columns
        assert Normal.x.dtype is UInt64

    def test_fallback_on_name_error_warns(self) -> None:
        """When get_type_hints raises NameError, fallback triggers with warning."""
        with (
            patch("colnade.schema.typing.get_type_hints", side_effect=NameError("bad ref")),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            class Fallback(Schema):
                val: Column[Utf8]

            assert len(w) == 1
            assert "get_type_hints() failed" in str(w[0].message)
            assert "Fallback" in str(w[0].message)

    def test_fallback_on_type_error_warns(self) -> None:
        """When get_type_hints raises TypeError, fallback triggers with warning."""
        with (
            patch("colnade.schema.typing.get_type_hints", side_effect=TypeError("bad type")),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            class Fallback2(Schema):
                val: Column[Utf8]

            assert len(w) == 1
            assert "get_type_hints() failed" in str(w[0].message)

    def test_fallback_still_creates_columns(self) -> None:
        """Fallback path still creates Column descriptors from raw annotations."""
        with (
            patch("colnade.schema.typing.get_type_hints", side_effect=NameError("x")),
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")

            class FallbackSchema(Schema):
                name: Column[Utf8]
                age: Column[UInt64]

            assert "name" in FallbackSchema._columns
            assert "age" in FallbackSchema._columns
            assert FallbackSchema.name.name == "name"

    def test_unexpected_exception_not_caught(self) -> None:
        """Unexpected exceptions (not NameError/TypeError) propagate."""
        with patch("colnade.schema.typing.get_type_hints", side_effect=RuntimeError("unexpected")):
            try:

                class BadSchema(Schema):
                    val: Column[Utf8]

                raise AssertionError("RuntimeError should have propagated")
            except RuntimeError:
                pass  # Expected


# ---------------------------------------------------------------------------
# Schema repr
# ---------------------------------------------------------------------------


class TestSchemaRepr:
    def test_repr_with_columns(self) -> None:
        r = repr(Users)
        assert r.startswith("Users(")
        assert "id: UInt64" in r
        assert "name: Utf8" in r

    def test_repr_empty_schema(self) -> None:
        assert repr(Empty) == "Empty"

    def test_repr_html_with_columns(self) -> None:
        html = Users._repr_html_()
        assert "<b>Users</b>" in html
        assert "<th>Column</th>" in html
        assert "<td>id</td>" in html
        assert "<td>UInt64</td>" in html

    def test_repr_html_empty_schema(self) -> None:
        assert Empty._repr_html_() == "<b>Empty</b>"

    def test_repr_inherited_schema(self) -> None:
        r = repr(EnrichedUsers)
        assert "normalized_age: Float64" in r
        assert "id: UInt64" in r


# ---------------------------------------------------------------------------
# Schema.Row auto-generated dataclass
# ---------------------------------------------------------------------------


class RowTestSchema(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    score: Column[Float64]
    data: Column[Binary]
    created: Column[Date]
    updated: Column[Datetime]


class NullableSchema(Schema):
    id: Column[UInt64]
    name: Column[Utf8 | None]


class TestSchemaRow:
    def test_row_exists(self) -> None:
        assert hasattr(Users, "Row")

    def test_row_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(Users.Row)

    def test_row_is_frozen(self) -> None:
        row = Users.Row(id=1, name="Alice", age=10, score=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.id = 2  # type: ignore[misc]

    def test_row_has_slots(self) -> None:
        assert hasattr(Users.Row, "__slots__")

    def test_row_class_name(self) -> None:
        assert Users.Row.__name__ == "UsersRow"

    def test_row_field_names_match_columns(self) -> None:
        fields = {f.name for f in dataclasses.fields(Users.Row)}
        assert fields == set(Users._columns.keys())

    def test_row_field_types_simple(self) -> None:
        field_types = {f.name: f.type for f in dataclasses.fields(RowTestSchema.Row)}
        assert field_types["id"] is int
        assert field_types["name"] is str
        assert field_types["score"] is float
        assert field_types["data"] is bytes
        assert field_types["created"] is datetime.date
        assert field_types["updated"] is datetime.datetime

    def test_row_nullable_field_type(self) -> None:
        field_types = {f.name: f.type for f in dataclasses.fields(NullableSchema.Row)}
        assert field_types["id"] is int
        # name should be str | None
        name_type = field_types["name"]
        assert isinstance(name_type, types.UnionType)
        assert str in name_type.__args__
        assert type(None) in name_type.__args__

    def test_row_construction(self) -> None:
        row = RowTestSchema.Row(
            id=42,
            name="Alice",
            score=3.14,
            data=b"hello",
            created=datetime.date(2024, 1, 1),
            updated=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        assert row.id == 42
        assert row.name == "Alice"
        assert row.score == 3.14

    def test_row_equality(self) -> None:
        r1 = NullableSchema.Row(id=1, name="Alice")
        r2 = NullableSchema.Row(id=1, name="Alice")
        assert r1 == r2

    def test_row_repr(self) -> None:
        row = NullableSchema.Row(id=1, name="Bob")
        r = repr(row)
        assert "NullableSchemaRow" in r
        assert "id=1" in r
        assert "name='Bob'" in r

    def test_inherited_schema_row_has_all_fields(self) -> None:
        assert hasattr(EnrichedUsers, "Row")
        fields = {f.name for f in dataclasses.fields(EnrichedUsers.Row)}
        assert fields == set(EnrichedUsers._columns.keys())

    def test_inherited_schema_row_class_name(self) -> None:
        assert EnrichedUsers.Row.__name__ == "EnrichedUsersRow"

    def test_empty_schema_no_row(self) -> None:
        assert not hasattr(Empty, "Row")

    def test_only_private_schema_no_row(self) -> None:
        assert not hasattr(OnlyPrivate, "Row")

    def test_row_missing_field_raises(self) -> None:
        with pytest.raises(TypeError):
            NullableSchema.Row(id=1)  # missing 'name'

    def test_row_extra_field_raises(self) -> None:
        with pytest.raises(TypeError):
            NullableSchema.Row(id=1, name="Alice", extra=99)

    def test_row_delete_attr_raises(self) -> None:
        row = NullableSchema.Row(id=1, name="Alice")
        with pytest.raises(AttributeError):
            del row.id
