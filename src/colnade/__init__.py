"""Colnade: A statically type-safe DataFrame abstraction layer."""

from colnade._protocols import BackendProtocol
from colnade.arrow import ArrowBatch
from colnade.constraints import Field, FieldInfo, ValueViolation, schema_check
from colnade.dataframe import (
    DataFrame,
    GroupBy,
    JoinedDataFrame,
    JoinedLazyFrame,
    LazyFrame,
    LazyGroupBy,
    UntypedDataFrame,
    UntypedLazyFrame,
)
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
from colnade.expr import (
    Agg,
    AliasedExpr,
    BinOp,
    ColumnRef,
    Expr,
    FunctionCall,
    JoinCondition,
    ListOp,
    Literal,
    SortExpr,
    StructFieldAccess,
    UnaryOp,
    lit,
)
from colnade.schema import Column, ListAccessor, Row, Schema, SchemaError, mapped_from
from colnade.validation import (
    ValidationLevel,
    get_validation_level,
    is_validation_enabled,
    set_validation,
)

__all__ = [
    # Backend
    "BackendProtocol",
    # Arrow boundary
    "ArrowBatch",
    # Schema layer
    "Schema",
    "Column",
    "Row",
    "ListAccessor",
    "mapped_from",
    "SchemaError",
    # DataFrame layer
    "DataFrame",
    "LazyFrame",
    "GroupBy",
    "LazyGroupBy",
    "JoinedDataFrame",
    "JoinedLazyFrame",
    "UntypedDataFrame",
    "UntypedLazyFrame",
    # Expression DSL
    "Expr",
    "ColumnRef",
    "BinOp",
    "UnaryOp",
    "Literal",
    "FunctionCall",
    "Agg",
    "AliasedExpr",
    "SortExpr",
    "StructFieldAccess",
    "ListOp",
    "JoinCondition",
    "lit",
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
    # Validation
    "ValidationLevel",
    "set_validation",
    "is_validation_enabled",
    "get_validation_level",
    # Constraints
    "Field",
    "FieldInfo",
    "ValueViolation",
    "schema_check",
]
