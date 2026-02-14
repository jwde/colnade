"""Runtime schema validation toggle.

Validation is **off** by default for zero overhead in production.
Enable it via environment variable (ideal for CI) or programmatically::

    # Environment variable
    COLNADE_VALIDATE=1 pytest tests/

    # Programmatic
    import colnade
    colnade.set_validation(True)

When enabled:

- Data boundaries (``read_parquet``, ``from_batches``, etc.) validate
  that loaded data matches the declared schema.
- Expression construction checks that literal values are type-compatible
  with the column dtype (e.g., ``float`` rejected for ``UInt8`` column).

``DataFrame.validate()`` and ``LazyFrame.validate()`` always run
explicitly regardless of this toggle.
"""

from __future__ import annotations

import datetime
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

_validation_enabled: bool | None = None


def is_validation_enabled() -> bool:
    """Return whether automatic validation at data boundaries is enabled."""
    if _validation_enabled is not None:
        return _validation_enabled
    return os.environ.get("COLNADE_VALIDATE", "").lower() in ("1", "true", "yes")


def set_validation(enabled: bool) -> None:
    """Enable or disable automatic validation at data boundaries."""
    global _validation_enabled
    _validation_enabled = enabled


# ---------------------------------------------------------------------------
# Dtype → Python type mapping for literal validation
# ---------------------------------------------------------------------------

_DTYPE_PYTHON_TYPES: dict[type, tuple[type, ...]] | None = None


def _get_dtype_python_types() -> dict[type, tuple[type, ...]]:
    """Lazy-init the dtype-to-Python-type mapping."""
    global _DTYPE_PYTHON_TYPES
    if _DTYPE_PYTHON_TYPES is not None:
        return _DTYPE_PYTHON_TYPES

    from colnade import dtypes

    _DTYPE_PYTHON_TYPES = {
        dtypes.Bool: (bool,),
        dtypes.UInt8: (int,),
        dtypes.UInt16: (int,),
        dtypes.UInt32: (int,),
        dtypes.UInt64: (int,),
        dtypes.Int8: (int,),
        dtypes.Int16: (int,),
        dtypes.Int32: (int,),
        dtypes.Int64: (int,),
        dtypes.Float32: (int, float),
        dtypes.Float64: (int, float),
        dtypes.Utf8: (str,),
        dtypes.Binary: (bytes,),
        dtypes.Date: (datetime.date,),
        dtypes.Time: (datetime.time,),
        dtypes.Datetime: (datetime.datetime,),
        dtypes.Duration: (datetime.timedelta,),
    }
    return _DTYPE_PYTHON_TYPES


# ---------------------------------------------------------------------------
# Dtype → Python type mapping for Row dataclass fields
# ---------------------------------------------------------------------------

_DTYPE_ROW_TYPES: dict[type, type] | None = None


def _get_dtype_row_types() -> dict[type, type]:
    """Lazy-init the dtype-to-Python-type mapping for Row generation."""
    global _DTYPE_ROW_TYPES
    if _DTYPE_ROW_TYPES is not None:
        return _DTYPE_ROW_TYPES

    from colnade import dtypes

    _DTYPE_ROW_TYPES = {
        dtypes.Bool: bool,
        dtypes.UInt8: int,
        dtypes.UInt16: int,
        dtypes.UInt32: int,
        dtypes.UInt64: int,
        dtypes.Int8: int,
        dtypes.Int16: int,
        dtypes.Int32: int,
        dtypes.Int64: int,
        dtypes.Float32: float,
        dtypes.Float64: float,
        dtypes.Utf8: str,
        dtypes.Binary: bytes,
        dtypes.Date: datetime.date,
        dtypes.Time: datetime.time,
        dtypes.Datetime: datetime.datetime,
        dtypes.Duration: datetime.timedelta,
    }
    return _DTYPE_ROW_TYPES


def dtype_to_python_type(dtype: Any) -> type:
    """Map a Colnade dtype annotation to a Python type for Row dataclass fields.

    Handles nullable unions (``UInt64 | None`` → ``int | None``),
    ``List[T]`` → ``list``, ``Struct[S]`` → ``dict``.
    Returns ``object`` for unrecognised dtypes.
    """
    import types as _types
    import typing

    is_nullable = False
    inner_dtype = dtype

    # Handle nullable unions: T | None
    if isinstance(dtype, _types.UnionType):
        args = [a for a in dtype.__args__ if a is not type(None)]
        if len(args) == 1:
            inner_dtype = args[0]
            is_nullable = True
        else:
            return object  # multi-type union, fallback

    # Handle parameterized nested types (List[T], Struct[S])
    inner_origin = typing.get_origin(inner_dtype)
    if inner_origin is not None:
        from colnade.dtypes import List, Struct

        if inner_origin is Struct:
            py_type: type = dict
        elif inner_origin is List:
            py_type = list
        else:
            py_type = object
    else:
        mapping = _get_dtype_row_types()
        py_type = mapping.get(inner_dtype, object)

    if is_nullable:
        return py_type | None  # type: ignore[return-value]
    return py_type


# ---------------------------------------------------------------------------
# Literal type checking
# ---------------------------------------------------------------------------


def check_literal_type(value: Any, dtype: Any, context: str = "") -> None:
    """Check that a Python literal is compatible with a Colnade dtype.

    Only runs when validation is enabled. Raises TypeError on mismatch.
    """
    if not is_validation_enabled():
        return

    if value is None:
        return

    import types

    # Skip nullable union types — any value compatible with the base type is fine
    if isinstance(dtype, types.UnionType):
        args = [a for a in dtype.__args__ if a is not type(None)]
        if len(args) == 1:
            dtype = args[0]
        else:
            return

    mapping = _get_dtype_python_types()
    allowed = mapping.get(dtype)
    if allowed is None:
        return  # Unknown dtype (Struct, List, etc.) — skip

    if not isinstance(value, allowed):
        allowed_names = ", ".join(t.__name__ for t in allowed)
        dtype_name = dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)
        ctx = f" in {context}" if context else ""
        msg = (
            f"Type mismatch{ctx}: got {type(value).__name__} value {value!r}, "
            f"expected {allowed_names} for dtype {dtype_name}"
        )
        raise TypeError(msg)
