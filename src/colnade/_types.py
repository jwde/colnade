"""Internal typing utilities for Colnade.

Core TypeVars used across the library. Schema-bound TypeVars (S, S2, S3,
SchemaType) are defined in schema.py once Schema exists.
"""

from __future__ import annotations

from typing import TypeVar

from colnade.dtypes import FloatType, NumericType

# General-purpose type variable for column data types
DType = TypeVar("DType")

# Element type (used in List[T] and other generic contexts)
T = TypeVar("T")

# Numeric type variable (for constraining methods like .sum(), .mean())
N = TypeVar("N", bound=NumericType)

# Float type variable (for constraining NaN methods like .is_nan(), .fill_nan())
F = TypeVar("F", bound=FloatType)
