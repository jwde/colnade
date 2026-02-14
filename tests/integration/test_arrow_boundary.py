"""Integration tests for Arrow boundary (to_batches / from_batches)."""

from __future__ import annotations

import polars as pl
import pytest

from colnade import ArrowBatch, Column, DataFrame, Schema, UInt64, Utf8
from colnade_polars.adapter import PolarsBackend

pa = pytest.importorskip("pyarrow")


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


def _users_df() -> DataFrame[Users]:
    data = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "name": ["Alice", "Bob", "Charlie"],
        }
    )
    return DataFrame(_data=data, _schema=Users, _backend=PolarsBackend())


# ---------------------------------------------------------------------------
# to_batches
# ---------------------------------------------------------------------------


class TestToBatches:
    def test_returns_arrow_batches(self) -> None:
        df = _users_df()
        batches = list(df.to_batches())
        assert len(batches) >= 1
        assert all(isinstance(b, ArrowBatch) for b in batches)

    def test_preserves_schema_type(self) -> None:
        df = _users_df()
        batches = list(df.to_batches())
        for batch in batches:
            assert batch.schema is Users

    def test_with_batch_size(self) -> None:
        df = _users_df()
        batches = list(df.to_batches(batch_size=2))
        assert len(batches) == 2
        assert batches[0].num_rows == 2
        assert batches[1].num_rows == 1

    def test_pyarrow_roundtrip(self) -> None:
        df = _users_df()
        batches = list(df.to_batches())
        for batch in batches:
            rb = batch.to_pyarrow()
            assert isinstance(rb, pa.RecordBatch)
            assert set(rb.schema.names) == {"id", "name"}

    def test_no_backend_raises(self) -> None:
        df: DataFrame[Users] = DataFrame(_schema=Users)
        with pytest.raises(RuntimeError, match="requires a backend"):
            list(df.to_batches())

    def test_total_rows_match(self) -> None:
        df = _users_df()
        batches = list(df.to_batches())
        total = sum(b.num_rows for b in batches)
        assert total == 3

    def test_batch_size_one(self) -> None:
        df = _users_df()
        batches = list(df.to_batches(batch_size=1))
        assert len(batches) == 3
        assert all(b.num_rows == 1 for b in batches)


# ---------------------------------------------------------------------------
# from_batches
# ---------------------------------------------------------------------------


class TestFromBatches:
    def test_round_trip(self) -> None:
        original = _users_df()
        batches = list(original.to_batches())
        restored = DataFrame.from_batches(iter(batches), Users, PolarsBackend())
        assert restored._data.shape == (3, 2)
        assert restored._data["name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert restored._data["id"].to_list() == [1, 2, 3]

    def test_preserves_schema(self) -> None:
        original = _users_df()
        batches = list(original.to_batches())
        restored = DataFrame.from_batches(iter(batches), Users, PolarsBackend())
        assert restored._schema is Users

    def test_preserves_backend(self) -> None:
        backend = PolarsBackend()
        original = _users_df()
        batches = list(original.to_batches())
        restored = DataFrame.from_batches(iter(batches), Users, backend)
        assert restored._backend is backend

    def test_multiple_batches(self) -> None:
        original = _users_df()
        batches = list(original.to_batches(batch_size=1))
        assert len(batches) == 3
        restored = DataFrame.from_batches(iter(batches), Users, PolarsBackend())
        assert restored._data.shape == (3, 2)

    def test_round_trip_data_integrity(self) -> None:
        original = _users_df()
        batches = list(original.to_batches(batch_size=2))
        restored = DataFrame.from_batches(iter(batches), Users, PolarsBackend())
        # Verify row-level data integrity
        orig_rows = original._data.sort("id").to_dicts()
        rest_rows = restored._data.sort("id").to_dicts()
        assert orig_rows == rest_rows


# ---------------------------------------------------------------------------
# PolarsBackend Arrow methods directly
# ---------------------------------------------------------------------------


class TestPolarsBackendArrowMethods:
    def test_to_arrow_batches_returns_record_batches(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame(
            {
                "id": pl.Series([1, 2], dtype=pl.UInt64),
                "name": ["a", "b"],
            }
        )
        batches = list(backend.to_arrow_batches(data, None))
        assert all(isinstance(b, pa.RecordBatch) for b in batches)

    def test_to_arrow_batches_with_batch_size(self) -> None:
        backend = PolarsBackend()
        data = pl.DataFrame(
            {
                "id": pl.Series(list(range(10)), dtype=pl.UInt64),
                "name": [f"u{i}" for i in range(10)],
            }
        )
        batches = list(backend.to_arrow_batches(data, 3))
        total = sum(b.num_rows for b in batches)
        assert total == 10

    def test_from_arrow_batches_returns_polars_df(self) -> None:
        backend = PolarsBackend()
        rb = pa.record_batch(
            {"id": [1, 2], "name": ["a", "b"]},
            schema=pa.schema([("id", pa.uint64()), ("name", pa.string())]),
        )
        result = backend.from_arrow_batches(iter([rb]), Users)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_from_arrow_batches_empty(self) -> None:
        backend = PolarsBackend()
        result = backend.from_arrow_batches(iter([]), Users)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (0, 0)

    def test_from_arrow_batches_multiple(self) -> None:
        backend = PolarsBackend()
        schema = pa.schema([("id", pa.uint64()), ("name", pa.string())])
        rb1 = pa.record_batch({"id": [1], "name": ["a"]}, schema=schema)
        rb2 = pa.record_batch({"id": [2], "name": ["b"]}, schema=schema)
        result = backend.from_arrow_batches(iter([rb1, rb2]), Users)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)
