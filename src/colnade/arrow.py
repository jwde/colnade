"""Typed Arrow batch wrapper for cross-framework boundaries.

ArrowBatch[S] wraps a pyarrow.RecordBatch while preserving the schema type
parameter, enabling type-safe data transfer between backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from colnade.schema import S, Schema, SchemaError

if TYPE_CHECKING:
    import pyarrow as pa


class ArrowBatch(Generic[S]):
    """A typed wrapper around a pyarrow.RecordBatch.

    Preserves the schema type parameter S across Arrow serialization
    boundaries, so that type checkers can verify schema consistency
    when data moves between backends.
    """

    __slots__ = ("_batch", "_schema")

    def __init__(
        self,
        *,
        _batch: Any,
        _schema: type[S],
    ) -> None:
        self._batch = _batch
        self._schema = _schema

    def __repr__(self) -> str:
        schema_name = self._schema.__name__ if self._schema else "Any"
        n_rows = self._batch.num_rows if self._batch is not None else 0
        return f"ArrowBatch[{schema_name}]({n_rows} rows)"

    def to_pyarrow(self) -> pa.RecordBatch:
        """Return the underlying pyarrow.RecordBatch."""
        return self._batch

    @classmethod
    def from_pyarrow(
        cls,
        batch: pa.RecordBatch,
        schema: type[S],
    ) -> ArrowBatch[S]:
        """Wrap a pyarrow.RecordBatch with schema validation.

        Checks that the Arrow batch's column names match the schema.
        Raises :class:`SchemaError` on missing columns.
        """
        _validate_arrow_schema(batch.schema, schema)
        return cls(_batch=batch, _schema=schema)

    @property
    def num_rows(self) -> int:
        """Number of rows in this batch."""
        return self._batch.num_rows

    @property
    def schema(self) -> type[S]:
        """The Colnade schema type."""
        return self._schema


def _validate_arrow_schema(
    arrow_schema: Any,
    colnade_schema: type[Schema],
) -> None:
    """Validate that an Arrow schema's columns match a Colnade schema.

    Checks column names are present. Type validation is left to backends
    since Arrow-to-Colnade dtype mapping is backend-specific.
    """
    arrow_names = set(arrow_schema.names) if arrow_schema is not None else set()
    expected = colnade_schema._columns
    missing = [name for name in expected if name not in arrow_names]
    if missing:
        raise SchemaError(missing_columns=missing)
