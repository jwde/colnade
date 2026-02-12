"""Data type definitions for Colnade.

All column types are sentinel classes — they don't hold data. They exist so that
Column[UInt8, S] is a meaningful generic type and the type checker can distinguish
column types and enforce method availability by dtype.

Type categories (NumericType, FloatType, TemporalType) are base classes, enabling
TypeVar bounds for method constraints (e.g., .sum() only on NumericType columns).
"""

from __future__ import annotations

from typing import Generic, TypeVar

# ---------------------------------------------------------------------------
# Type category base classes
# ---------------------------------------------------------------------------


class NumericType:
    """Base class for all numeric data types."""


class IntegerType(NumericType):
    """Base class for all integer data types (signed and unsigned)."""


class FloatType(NumericType):
    """Base class for floating-point data types. Used to constrain NaN methods."""


class TemporalType:
    """Base class for all temporal data types."""


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


class Bool:
    """Boolean type. Not numeric — arithmetic on booleans is not supported."""


# ---------------------------------------------------------------------------
# Unsigned integers
# ---------------------------------------------------------------------------


class UInt8(IntegerType):
    """8-bit unsigned integer."""


class UInt16(IntegerType):
    """16-bit unsigned integer."""


class UInt32(IntegerType):
    """32-bit unsigned integer."""


class UInt64(IntegerType):
    """64-bit unsigned integer."""


# ---------------------------------------------------------------------------
# Signed integers
# ---------------------------------------------------------------------------


class Int8(IntegerType):
    """8-bit signed integer."""


class Int16(IntegerType):
    """16-bit signed integer."""


class Int32(IntegerType):
    """32-bit signed integer."""


class Int64(IntegerType):
    """64-bit signed integer."""


# ---------------------------------------------------------------------------
# Floating point
# ---------------------------------------------------------------------------


class Float32(FloatType):
    """32-bit floating point."""


class Float64(FloatType):
    """64-bit floating point."""


# ---------------------------------------------------------------------------
# String / Binary
# ---------------------------------------------------------------------------


class Utf8:
    """UTF-8 encoded string."""


class Binary:
    """Raw binary data."""


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


class Date(TemporalType):
    """Calendar date (no time component)."""


class Time(TemporalType):
    """Time of day (no date component)."""


class Datetime(TemporalType):
    """Date and time."""


class Duration(TemporalType):
    """Time duration / interval."""


# ---------------------------------------------------------------------------
# Parameterized nested types
# ---------------------------------------------------------------------------

_S = TypeVar("_S")
_T = TypeVar("_T")


class Struct(Generic[_S]):
    """A struct column type parameterized by a Schema class.

    Usage in schema definitions::

        class Address(Schema):
            street: Utf8
            city: Utf8

        class Users(Schema):
            address: Struct[Address]
            location: Struct[GeoPoint] | None  # nullable struct
    """


class List(Generic[_T]):
    """A list column type parameterized by element type.

    Usage in schema definitions::

        class Users(Schema):
            tags: List[Utf8]                  # list of strings
            scores: List[Float64 | None]      # list of nullable floats
            friends: List[UInt64] | None      # nullable list
    """
