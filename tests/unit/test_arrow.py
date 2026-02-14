"""Unit tests for ArrowBatch[S] typed wrapper."""

from __future__ import annotations

import pytest

from colnade import ArrowBatch, Column, Schema, SchemaError, UInt64, Utf8


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------


class TestArrowBatchConstruction:
    def test_init_stores_batch_and_schema(self) -> None:
        sentinel = object()
        batch = ArrowBatch(_batch=sentinel, _schema=Users)
        assert batch._batch is sentinel
        assert batch._schema is Users

    def test_repr_includes_schema_name(self) -> None:
        class FakeBatch:
            num_rows = 42

        batch = ArrowBatch(_batch=FakeBatch(), _schema=Users)
        r = repr(batch)
        assert "Users" in r
        assert "42" in r

    def test_to_pyarrow_returns_inner(self) -> None:
        sentinel = object()
        batch = ArrowBatch(_batch=sentinel, _schema=Users)
        assert batch.to_pyarrow() is sentinel

    def test_schema_property(self) -> None:
        batch = ArrowBatch(_batch=None, _schema=Users)
        assert batch.schema is Users


# ---------------------------------------------------------------------------
# from_pyarrow (requires pyarrow)
# ---------------------------------------------------------------------------


class TestArrowBatchFromPyarrow:
    @pytest.fixture(autouse=True)
    def _skip_no_pyarrow(self) -> None:
        pytest.importorskip("pyarrow")

    def test_from_pyarrow_valid(self) -> None:
        import pyarrow as pa

        rb = pa.record_batch(
            {"id": [1, 2], "name": ["Alice", "Bob"]},
            schema=pa.schema([("id", pa.uint64()), ("name", pa.string())]),
        )
        batch = ArrowBatch.from_pyarrow(rb, Users)
        assert batch.num_rows == 2
        assert batch.schema is Users

    def test_from_pyarrow_missing_column_raises(self) -> None:
        import pyarrow as pa

        rb = pa.record_batch(
            {"id": [1, 2]},
            schema=pa.schema([("id", pa.uint64())]),
        )
        with pytest.raises(SchemaError):
            ArrowBatch.from_pyarrow(rb, Users)

    def test_from_pyarrow_extra_columns_ok(self) -> None:
        import pyarrow as pa

        rb = pa.record_batch(
            {"id": [1], "name": ["Alice"], "extra": [99]},
            schema=pa.schema(
                [
                    ("id", pa.uint64()),
                    ("name", pa.string()),
                    ("extra", pa.int64()),
                ]
            ),
        )
        batch = ArrowBatch.from_pyarrow(rb, Users)
        assert batch.num_rows == 1

    def test_num_rows_property(self) -> None:
        import pyarrow as pa

        rb = pa.record_batch(
            {"id": [1, 2, 3], "name": ["a", "b", "c"]},
            schema=pa.schema([("id", pa.uint64()), ("name", pa.string())]),
        )
        batch = ArrowBatch.from_pyarrow(rb, Users)
        assert batch.num_rows == 3
