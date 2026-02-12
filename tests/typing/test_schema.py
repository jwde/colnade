"""Static type tests for colnade.schema.

This file is checked by ty — it must produce zero type errors.
It verifies that Schema, Column, and the type hierarchy are
visible to the type checker.
"""

from colnade import Column, Datetime, Float64, Schema, UInt8, UInt64, Utf8

# --- Schema definition compiles cleanly ---


class Users(Schema):
    id: UInt64
    name: Utf8
    age: UInt8 | None
    score: Float64


class EnrichedUsers(Users):
    normalized_age: Float64


class HasUserId(Schema):
    user_id: UInt64


class HasTimestamp(Schema):
    created_at: Datetime


class Events(HasUserId, HasTimestamp):
    event_type: Utf8


# --- Column access produces Column instances ---


def check_column_access() -> None:
    _id: Column[UInt64, Users] = Users.id  # type: ignore[assignment]
    _name: Column[Utf8, Users] = Users.name  # type: ignore[assignment]
    _ = _id
    _ = _name


# --- Schema and Column are importable and usable as types ---


def check_types_exist() -> None:
    _: type[Schema] = Schema
    _: type[Column[UInt64, Users]] = Column


# --- Schema-bound TypeVars are importable ---


def check_schema_typevars() -> None:
    from colnade.schema import S2, S3, S, SchemaType

    _ = S
    _ = S2
    _ = S3
    _ = SchemaType
