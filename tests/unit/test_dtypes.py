"""Unit tests for colnade.dtypes and colnade._types."""

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

# ---------------------------------------------------------------------------
# Type category hierarchy
# ---------------------------------------------------------------------------


class TestTypeHierarchy:
    """Verify the inheritance relationships between dtype categories."""

    def test_integer_types_are_numeric(self) -> None:
        for cls in (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64):
            assert issubclass(cls, NumericType), f"{cls.__name__} should be NumericType"
            assert issubclass(cls, IntegerType), f"{cls.__name__} should be IntegerType"

    def test_float_types_are_numeric(self) -> None:
        for cls in (Float32, Float64):
            assert issubclass(cls, NumericType), f"{cls.__name__} should be NumericType"
            assert issubclass(cls, FloatType), f"{cls.__name__} should be FloatType"

    def test_float_types_are_not_integer(self) -> None:
        for cls in (Float32, Float64):
            assert not issubclass(cls, IntegerType), f"{cls.__name__} should not be IntegerType"

    def test_integer_types_are_not_float(self) -> None:
        for cls in (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64):
            assert not issubclass(cls, FloatType), f"{cls.__name__} should not be FloatType"

    def test_temporal_types(self) -> None:
        for cls in (Date, Time, Datetime, Duration):
            assert issubclass(cls, TemporalType), f"{cls.__name__} should be TemporalType"
            assert not issubclass(cls, NumericType), f"{cls.__name__} should not be NumericType"

    def test_bool_is_standalone(self) -> None:
        assert not issubclass(Bool, NumericType)
        assert not issubclass(Bool, TemporalType)

    def test_utf8_is_standalone(self) -> None:
        assert not issubclass(Utf8, NumericType)
        assert not issubclass(Utf8, TemporalType)

    def test_binary_is_standalone(self) -> None:
        assert not issubclass(Binary, NumericType)
        assert not issubclass(Binary, TemporalType)


# ---------------------------------------------------------------------------
# Sentinel classes are instantiable (needed for runtime introspection)
# ---------------------------------------------------------------------------


class TestSentinelInstantiation:
    """Sentinel dtype classes should be simple instantiable classes."""

    def test_all_dtypes_instantiable(self) -> None:
        all_dtypes = [
            Bool,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Int8,
            Int16,
            Int32,
            Int64,
            Float32,
            Float64,
            Utf8,
            Binary,
            Date,
            Time,
            Datetime,
            Duration,
        ]
        for cls in all_dtypes:
            instance = cls()
            assert isinstance(instance, cls)


# ---------------------------------------------------------------------------
# Parameterized nested types
# ---------------------------------------------------------------------------


class TestParameterizedTypes:
    """Struct and List are Generic types usable as type parameters."""

    def test_struct_is_generic(self) -> None:
        # Struct[X] should be subscriptable at runtime
        alias = Struct[int]
        assert alias is not None

    def test_list_is_generic(self) -> None:
        # List[X] should be subscriptable at runtime
        alias = List[int]
        assert alias is not None


# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify __init__.py re-exports all expected names."""

    def test_all_types_in_colnade_namespace(self) -> None:
        import colnade

        expected = [
            "NumericType",
            "IntegerType",
            "FloatType",
            "TemporalType",
            "Bool",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Float32",
            "Float64",
            "Utf8",
            "Binary",
            "Date",
            "Time",
            "Datetime",
            "Duration",
            "Struct",
            "List",
        ]
        for name in expected:
            assert hasattr(colnade, name), f"colnade.{name} should be exported"

    def test_all_list_complete(self) -> None:
        import colnade

        # __all__ should contain every expected public name
        expected = {
            "NumericType",
            "IntegerType",
            "FloatType",
            "TemporalType",
            "Bool",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Float32",
            "Float64",
            "Utf8",
            "Binary",
            "Date",
            "Time",
            "Datetime",
            "Duration",
            "Struct",
            "List",
        }
        assert expected == set(colnade.__all__)


# ---------------------------------------------------------------------------
# TypeVars (basic sanity — they're just TypeVar objects)
# ---------------------------------------------------------------------------


class TestTypeVars:
    """Verify TypeVars exist and have correct bounds."""

    def test_dtype_typevar(self) -> None:
        from colnade._types import DType

        assert DType.__name__ == "DType"

    def test_numeric_typevar_bound(self) -> None:
        from colnade._types import N

        assert N.__bound__ is NumericType

    def test_float_typevar_bound(self) -> None:
        from colnade._types import F

        assert F.__bound__ is FloatType

    def test_element_typevar(self) -> None:
        from colnade._types import T

        assert T.__name__ == "T"
