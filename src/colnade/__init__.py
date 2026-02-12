"""Colnade: A statically type-safe DataFrame abstraction layer."""

from colnade.dtypes import (
    Binary,
    Bool,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    FloatType,
    Int8,
    Int16,
    Int32,
    Int64,
    IntegerType,
    List,
    NumericType,
    Struct,
    TemporalType,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)
from colnade.schema import Column, Schema

__all__ = [
    # Schema layer
    "Schema",
    "Column",
    # Type categories
    "NumericType",
    "IntegerType",
    "FloatType",
    "TemporalType",
    # Boolean
    "Bool",
    # Unsigned integers
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    # Signed integers
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    # Floating point
    "Float32",
    "Float64",
    # String / binary
    "Utf8",
    "Binary",
    # Temporal
    "Date",
    "Time",
    "Datetime",
    "Duration",
    # Parameterized nested types
    "Struct",
    "List",
]
