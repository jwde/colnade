"""Unit tests for colnade.schema — Schema, Column, SchemaMeta."""

from __future__ import annotations

import types

from colnade import (
    Column,
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
