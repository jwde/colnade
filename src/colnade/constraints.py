"""Value-level constraints for schema columns.

Provides:
- ``FieldInfo`` — stores constraints declared via ``Field()``
- ``ValueViolation`` — describes a single constraint failure
- ``schema_check`` — decorator for cross-column constraint methods
- ``Field()`` — constructor function with pydantic-compatible parameter names
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# FieldInfo — stores all constraints for a single column
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FieldInfo:
    """Immutable container for column-level value constraints.

    Created by ``Field()`` and attached to Column descriptors by
    ``SchemaMeta``.  Uses pydantic-compatible parameter names for
    familiarity.
    """

    # Numeric / temporal range bounds
    ge: Any | None = None
    gt: Any | None = None
    le: Any | None = None
    lt: Any | None = None

    # String length bounds
    min_length: int | None = None
    max_length: int | None = None

    # Regex pattern (string columns)
    pattern: str | None = None

    # Uniqueness
    unique: bool = False

    # Allowed value set
    isin: Sequence[Any] | None = None

    # Optional mapped_from source (for cast_schema integration)
    mapped_from: Any | None = None  # Column[DType] at runtime

    def has_constraints(self) -> bool:
        """Return True if any value constraint is set (excluding mapped_from)."""
        return (
            self.ge is not None
            or self.gt is not None
            or self.le is not None
            or self.lt is not None
            or self.min_length is not None
            or self.max_length is not None
            or self.pattern is not None
            or self.unique
            or self.isin is not None
        )

    def __post_init__(self) -> None:
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern {self.pattern!r}: {e}") from e
        if self.ge is not None and self.gt is not None:
            raise ValueError("Cannot specify both 'ge' and 'gt'")
        if self.le is not None and self.lt is not None:
            raise ValueError("Cannot specify both 'le' and 'lt'")


# ---------------------------------------------------------------------------
# ValueViolation — a single constraint failure
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ValueViolation:
    """Describes a value-level constraint failure for error reporting."""

    column: str
    constraint: str
    got_count: int
    sample_values: list[Any]


# ---------------------------------------------------------------------------
# SchemaCheck — cross-column constraint wrapper
# ---------------------------------------------------------------------------


class SchemaCheck:
    """Wrapper for cross-column constraint methods declared with ``@schema_check``."""

    __slots__ = ("name", "fn")

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.name = fn.__name__
        self.fn = fn

    def __repr__(self) -> str:
        return f"SchemaCheck({self.name!r})"


def schema_check(fn: Callable[..., Any]) -> SchemaCheck:
    """Decorator for declaring cross-column constraints on a Schema.

    The decorated method receives the schema class and must return
    a boolean expression (``Expr[Bool]``) that should be True for valid rows::

        class Events(Schema):
            start_date: Column[Date]
            end_date: Column[Date]

            @schema_check
            def dates_ordered(cls):
                return Events.start_date <= Events.end_date
    """
    return SchemaCheck(fn)


# ---------------------------------------------------------------------------
# Field() constructor function
# ---------------------------------------------------------------------------


def Field(
    *,
    ge: Any | None = None,
    gt: Any | None = None,
    le: Any | None = None,
    lt: Any | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    unique: bool = False,
    isin: Sequence[Any] | None = None,
    mapped_from: Any | None = None,
) -> Any:
    """Declare value-level constraints (and optional mapped_from) for a column.

    Returns a ``FieldInfo`` instance that ``SchemaMeta`` detects at class
    creation time.  The return type is ``Any`` so it satisfies the
    ``Column[DType]`` annotation (same pattern as ``mapped_from()``).

    Usage::

        class Users(Schema):
            age: Column[UInt64] = Field(ge=0, le=150)
            email: Column[Utf8] = Field(pattern=r"^[^@]+@[^@]+\\\\.[^@]+$")
            id: Column[UInt64] = Field(unique=True)
            status: Column[Utf8] = Field(isin=["active", "inactive"])
    """
    return FieldInfo(
        ge=ge,
        gt=gt,
        le=le,
        lt=lt,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        unique=unique,
        isin=isin,
        mapped_from=mapped_from,
    )


# ---------------------------------------------------------------------------
# Helpers for backend adapters
# ---------------------------------------------------------------------------


def get_column_constraints(schema: type) -> dict[str, FieldInfo]:
    """Extract columns that have FieldInfo constraints from a schema."""
    result: dict[str, FieldInfo] = {}
    for col_name, col in schema._columns.items():
        if col._field_info is not None and col._field_info.has_constraints():
            result[col_name] = col._field_info
    return result


def get_schema_checks(schema: type) -> list[SchemaCheck]:
    """Return the list of ``@schema_check`` methods on a schema."""
    return getattr(schema, "_schema_checks", [])
