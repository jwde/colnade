"""Static type tests for colnade.dtypes.

This file is checked by ty — it must produce zero type errors.
It verifies that the type hierarchy is visible to the type checker
and that generic parameterization works correctly.
"""

from colnade._types import DType, F, N, T
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

# --- Type category hierarchy visible to type checker ---


def check_integer_is_numeric(x: IntegerType) -> NumericType:
    return x


def check_float_is_numeric(x: FloatType) -> NumericType:
    return x


def check_uint8_is_integer(x: UInt8) -> IntegerType:
    return x


def check_int64_is_integer(x: Int64) -> IntegerType:
    return x


def check_float64_is_float(x: Float64) -> FloatType:
    return x


def check_date_is_temporal(x: Date) -> TemporalType:
    return x


def check_datetime_is_temporal(x: Datetime) -> TemporalType:
    return x


def check_duration_is_temporal(x: Duration) -> TemporalType:
    return x


def check_time_is_temporal(x: Time) -> TemporalType:
    return x


# --- Generic parameterized types ---

_struct_alias: type[Struct[int]] = Struct
_list_alias: type[List[str]] = List


# --- All sentinel classes exist and are usable as types ---


def check_all_types_exist() -> None:
    _: type[Bool] = Bool
    _: type[UInt8] = UInt8
    _: type[UInt16] = UInt16
    _: type[UInt32] = UInt32
    _: type[UInt64] = UInt64
    _: type[Int8] = Int8
    _: type[Int16] = Int16
    _: type[Int32] = Int32
    _: type[Int64] = Int64
    _: type[Float32] = Float32
    _: type[Float64] = Float64
    _: type[Utf8] = Utf8
    _: type[Binary] = Binary
    _: type[Date] = Date
    _: type[Time] = Time
    _: type[Datetime] = Datetime
    _: type[Duration] = Duration


# --- TypeVars are importable ---


def check_typevars_exist() -> None:
    _ = DType
    _ = T
    _ = N
    _ = F
