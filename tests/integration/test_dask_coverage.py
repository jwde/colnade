"""Tests targeting uncovered code paths in the Dask backend adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest

import colnade
from colnade import (
    Column,
    DataFrame,
    Datetime,
    Float64,
    List,
    Schema,
    SchemaError,
    UInt64,
    Utf8,
    ValidationLevel,
)
from colnade.constraints import Field
from colnade_dask.adapter import DaskBackend
from colnade_dask.io import read_csv, read_parquet, write_csv, write_parquet

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class NullableUsers(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64 | None]


class ConstrainedUsers(Schema):
    id: Column[UInt64] = Field(unique=True)
    name: Column[Utf8]
    age: Column[UInt64] = Field(ge=0, le=150)


class Scores(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    score: Column[Float64]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_backend = DaskBackend()


def _users_ddf() -> DataFrame[Users]:
    data = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3, 4, 5], dtype=pd.UInt64Dtype()),
            "name": pd.array(["Alice", "Bob", "Charlie", "Diana", "Eve"], dtype=pd.StringDtype()),
            "age": pd.array([30, 25, 35, 28, 40], dtype=pd.UInt64Dtype()),
        }
    )
    ddf = dd.from_pandas(data, npartitions=2)
    return DataFrame(_data=ddf, _schema=Users, _backend=_backend)


def _scores_ddf() -> DataFrame[Scores]:
    data = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
            "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
            "score": pd.array([85.5, 92.3, 78.1], dtype="Float64"),
        }
    )
    ddf = dd.from_pandas(data, npartitions=2)
    return DataFrame(_data=ddf, _schema=Scores, _backend=_backend)


# ---------------------------------------------------------------------------
# String method expressions
# ---------------------------------------------------------------------------


class TestStringMethods:
    def test_str_contains(self) -> None:
        df = _users_ddf()
        result = df.filter(Users.name.str_contains("li"))
        assert set(result._data.compute()["name"].tolist()) == {"Alice", "Charlie"}

    def test_str_starts_with(self) -> None:
        df = _users_ddf()
        result = df.filter(Users.name.str_starts_with("A"))
        assert result._data.compute()["name"].tolist() == ["Alice"]

    def test_str_ends_with(self) -> None:
        df = _users_ddf()
        result = df.filter(Users.name.str_ends_with("e"))
        names = set(result._data.compute()["name"].tolist())
        assert names == {"Alice", "Charlie", "Eve"}

    def test_str_to_lowercase(self) -> None:
        df = _users_ddf()
        result = df.with_columns(Users.name.str_to_lowercase().alias(Users.name))
        assert result._data.compute()["name"].tolist() == [
            "alice",
            "bob",
            "charlie",
            "diana",
            "eve",
        ]

    def test_str_to_uppercase(self) -> None:
        df = _users_ddf()
        result = df.with_columns(Users.name.str_to_uppercase().alias(Users.name))
        assert result._data.compute()["name"].tolist() == [
            "ALICE",
            "BOB",
            "CHARLIE",
            "DIANA",
            "EVE",
        ]

    def test_str_strip(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2], dtype=pd.UInt64Dtype()),
                "name": pd.array(["  Alice  ", "  Bob  "], dtype=pd.StringDtype()),
                "age": pd.array([30, 25], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Users, _backend=_backend)
        result = df.with_columns(Users.name.str_strip().alias(Users.name))
        assert result._data.compute()["name"].tolist() == ["Alice", "Bob"]

    def test_str_replace(self) -> None:
        df = _users_ddf()
        result = df.with_columns(Users.name.str_replace("li", "LI").alias(Users.name))
        assert result._data.compute()["name"].tolist() == [
            "ALIce",
            "Bob",
            "CharLIe",
            "Diana",
            "Eve",
        ]

    def test_str_len(self) -> None:
        df = _users_ddf()
        result = df.with_columns(Users.name.str_len().alias(Users.age))
        lengths = result._data.compute()["age"].tolist()
        assert lengths == [5, 3, 7, 5, 3]


# ---------------------------------------------------------------------------
# Null handling expressions
# ---------------------------------------------------------------------------


class TestNullHandling:
    def test_fill_null(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
                "age": pd.array([30, pd.NA, 35], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=NullableUsers, _backend=_backend)
        result = df.with_columns(NullableUsers.age.fill_null(0).alias(NullableUsers.age))
        assert result._data.compute()["age"].tolist() == [30, 0, 35]

    def test_fill_nan(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
                "score": pd.array([85.5, float("nan"), 78.1], dtype="Float64"),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Scores, _backend=_backend)
        result = df.with_columns(Scores.score.fill_nan(0.0).alias(Scores.score))
        scores = result._data.compute()["score"].tolist()
        assert scores[1] == 0.0


# ---------------------------------------------------------------------------
# I/O with validation
# ---------------------------------------------------------------------------


class TestIOValidation:
    def test_read_parquet_with_structural_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.STRUCTURAL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.parquet")
                df = _users_ddf()
                write_parquet(df, path)
                result = read_parquet(path, Users)
                assert result._data.compute().shape[0] == 5
        finally:
            colnade.set_validation(prev)

    def test_read_parquet_with_full_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.FULL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.parquet")
                df = _users_ddf()
                write_parquet(df, path)
                result = read_parquet(path, Users)
                assert result._data.compute().shape[0] == 5
        finally:
            colnade.set_validation(prev)

    def test_read_csv_with_structural_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.STRUCTURAL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.csv")
                df = _users_ddf()
                write_csv(df, path)
                result = read_csv(path, Users)
                assert result._data.compute().shape[0] == 5
        finally:
            colnade.set_validation(prev)

    def test_read_csv_with_full_validation(self) -> None:
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.FULL)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.csv")
                df = _users_ddf()
                write_csv(df, path)
                result = read_csv(path, Users)
                assert result._data.compute().shape[0] == 5
        finally:
            colnade.set_validation(prev)


# ---------------------------------------------------------------------------
# NumPy dtype compatibility (parquet written without extension types)
# ---------------------------------------------------------------------------


class TestNumpyDtypeCompat:
    """Validate that data with NumPy dtypes passes structural validation."""

    def test_validate_numpy_dtypes(self) -> None:
        """NumPy uint64/float64 should be accepted for UInt64/Float64 columns."""
        import numpy as np

        data = pd.DataFrame({
            "id": np.array([1, 2, 3], dtype=np.uint64),
            "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
            "score": np.array([85.5, 92.3, 78.1], dtype=np.float64),
        })
        ddf = dd.from_pandas(data, npartitions=1)
        # Should not raise
        _backend.validate_schema(ddf, Scores)

    def test_validate_extension_dtypes_still_work(self) -> None:
        """Extension types should still pass (regression guard)."""
        data = pd.DataFrame({
            "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
            "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
            "score": pd.array([85.5, 92.3, 78.1], dtype=pd.Float64Dtype()),
        })
        ddf = dd.from_pandas(data, npartitions=1)
        _backend.validate_schema(ddf, Scores)

    def test_read_parquet_numpy_dtypes(self) -> None:
        """Parquet written with plain NumPy dtypes should pass validation."""
        prev = colnade.get_validation_level()
        try:
            colnade.set_validation(ValidationLevel.STRUCTURAL)
            import numpy as np

            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "test.parquet")
                pd.DataFrame({
                    "id": np.array([1, 2, 3], dtype=np.uint64),
                    "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
                    "score": np.array([85.5, 92.3, 78.1], dtype=np.float64),
                }).to_parquet(path, index=False)
                result = read_parquet(path, Scores)
                assert result._data.compute().shape[0] == 3
        finally:
            colnade.set_validation(prev)


# ---------------------------------------------------------------------------
# Nullable column validation (skip for UnionType)
# ---------------------------------------------------------------------------


class TestNullableValidation:
    def test_nullable_column_allows_nulls(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Bob"], dtype=pd.StringDtype()),
                "age": pd.array([30, pd.NA], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=NullableUsers, _backend=_backend)
        result = df.validate()
        assert result is df

    def test_non_nullable_column_rejects_nulls(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Bob"], dtype=pd.StringDtype()),
                "age": pd.array([30, pd.NA], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Users, _backend=_backend)
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        assert "age" in exc_info.value.null_violations


# ---------------------------------------------------------------------------
# Arrow batch conversion
# ---------------------------------------------------------------------------


class TestArrowBatches:
    def test_to_arrow_batches(self) -> None:
        df = _users_ddf()
        batches = list(_backend.to_arrow_batches(df._data, batch_size=None))
        assert len(batches) >= 1
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 5

    def test_to_arrow_batches_with_batch_size(self) -> None:
        df = _users_ddf()
        batches = list(_backend.to_arrow_batches(df._data, batch_size=2))
        assert len(batches) >= 2
        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == 5

    def test_from_arrow_batches(self) -> None:
        df = _users_ddf()
        batches = list(_backend.to_arrow_batches(df._data, batch_size=None))
        result = _backend.from_arrow_batches(iter(batches), Users)
        assert result.compute().shape[0] == 5

    def test_from_arrow_batches_empty(self) -> None:
        result = _backend.from_arrow_batches(iter([]), Users)
        assert result.compute().shape[0] == 0

    def test_roundtrip(self) -> None:
        df = _users_ddf()
        batches = list(_backend.to_arrow_batches(df._data, batch_size=2))
        result = _backend.from_arrow_batches(iter(batches), Users)
        computed = result.compute()
        assert computed.shape[0] == 5
        assert set(computed["name"].tolist()) == {"Alice", "Bob", "Charlie", "Diana", "Eve"}


# ---------------------------------------------------------------------------
# Temporal method expressions
# ---------------------------------------------------------------------------


class Events(Schema):
    id: Column[UInt64]
    ts: Column[Datetime]


class TestTemporalMethods:
    @pytest.fixture()
    def events_ddf(self) -> DataFrame[Events]:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
                "ts": pd.array(
                    [
                        pd.Timestamp("2025-03-15 10:30:45"),
                        pd.Timestamp("2025-06-20 14:15:00"),
                        pd.Timestamp("2025-12-01 08:00:30"),
                    ],
                    dtype=pd.ArrowDtype(pa.timestamp("us")),
                ),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        return DataFrame(_data=ddf, _schema=Events, _backend=_backend)

    def test_dt_year(self, events_ddf: DataFrame[Events]) -> None:
        result = events_ddf.with_columns(Events.ts.dt_year().alias(Events.id))
        assert result._data.compute()["id"].tolist() == [2025, 2025, 2025]

    def test_dt_month(self, events_ddf: DataFrame[Events]) -> None:
        result = events_ddf.with_columns(Events.ts.dt_month().alias(Events.id))
        assert result._data.compute()["id"].tolist() == [3, 6, 12]

    def test_dt_day(self, events_ddf: DataFrame[Events]) -> None:
        result = events_ddf.with_columns(Events.ts.dt_day().alias(Events.id))
        assert result._data.compute()["id"].tolist() == [15, 20, 1]

    def test_dt_hour(self, events_ddf: DataFrame[Events]) -> None:
        result = events_ddf.with_columns(Events.ts.dt_hour().alias(Events.id))
        assert result._data.compute()["id"].tolist() == [10, 14, 8]

    def test_dt_minute(self, events_ddf: DataFrame[Events]) -> None:
        result = events_ddf.with_columns(Events.ts.dt_minute().alias(Events.id))
        assert result._data.compute()["id"].tolist() == [30, 15, 0]

    def test_dt_second(self, events_ddf: DataFrame[Events]) -> None:
        result = events_ddf.with_columns(Events.ts.dt_second().alias(Events.id))
        assert result._data.compute()["id"].tolist() == [45, 0, 30]


# ---------------------------------------------------------------------------
# List operations (via object columns in Dask)
# ---------------------------------------------------------------------------


class TaggedItems(Schema):
    id: Column[UInt64]
    tags: Column[List[Utf8]]
    scores: Column[List[Float64]]


class TestListOperations:
    @pytest.fixture()
    def list_ddf(self) -> DataFrame[TaggedItems]:
        # Dask converts plain-object list columns to string[pyarrow],
        # so use Arrow-backed list types to preserve list semantics.
        tags_arr = pa.array([["python", "data"], ["rust", "systems"], ["python", "ml", "data"]])
        scores_arr = pa.array([[85.0, 90.0], [92.5, 88.0, 95.0], [78.0]])
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
                "tags": pd.array(tags_arr, dtype=pd.ArrowDtype(pa.list_(pa.string()))),
                "scores": pd.array(scores_arr, dtype=pd.ArrowDtype(pa.list_(pa.float64()))),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        return DataFrame(_data=ddf, _schema=TaggedItems, _backend=_backend)

    def test_list_len(self, list_ddf: DataFrame[TaggedItems]) -> None:
        result = list_ddf.with_columns(TaggedItems.tags.list.len().alias(TaggedItems.id))
        assert result._data.compute()["id"].tolist() == [2, 2, 3]

    def test_list_get(self, list_ddf: DataFrame[TaggedItems]) -> None:
        result = list_ddf.with_columns(TaggedItems.tags.list.get(0).alias(TaggedItems.tags))
        assert result._data.compute()["tags"].tolist() == ["python", "rust", "python"]

    def test_list_contains(self, list_ddf: DataFrame[TaggedItems]) -> None:
        result = list_ddf.filter(TaggedItems.tags.list.contains("python"))
        assert result._data.compute().shape[0] == 2

    def test_list_sum(self, list_ddf: DataFrame[TaggedItems]) -> None:
        result = list_ddf.with_columns(TaggedItems.scores.list.sum().alias(TaggedItems.scores))
        assert result._data.compute()["scores"].tolist() == [175.0, 275.5, 78.0]

    def test_list_mean(self, list_ddf: DataFrame[TaggedItems]) -> None:
        result = list_ddf.with_columns(TaggedItems.scores.list.mean().alias(TaggedItems.scores))
        means = result._data.compute()["scores"].tolist()
        assert abs(means[0] - 87.5) < 0.01

    def test_list_min(self, list_ddf: DataFrame[TaggedItems]) -> None:
        result = list_ddf.with_columns(TaggedItems.scores.list.min().alias(TaggedItems.scores))
        assert result._data.compute()["scores"].tolist() == [85.0, 88.0, 78.0]

    def test_list_max(self, list_ddf: DataFrame[TaggedItems]) -> None:
        result = list_ddf.with_columns(TaggedItems.scores.list.max().alias(TaggedItems.scores))
        assert result._data.compute()["scores"].tolist() == [90.0, 95.0, 78.0]


# ---------------------------------------------------------------------------
# Adapter error branches
# ---------------------------------------------------------------------------


class TestAdapterErrorBranches:
    def test_unsupported_binop_raises(self) -> None:
        from colnade.expr import BinOp, ColumnRef, Literal

        bad_expr = BinOp(left=ColumnRef(column=Users.id), right=Literal(value=1), op="^^^")
        with pytest.raises(ValueError, match="Unsupported BinOp"):
            _backend.translate_expr(bad_expr)

    def test_unsupported_unaryop_raises(self) -> None:
        from colnade.expr import ColumnRef, UnaryOp

        bad_expr = UnaryOp(operand=ColumnRef(column=Users.id), op="bad_op")
        with pytest.raises(ValueError, match="Unsupported UnaryOp"):
            _backend.translate_expr(bad_expr)

    def test_unsupported_agg_raises(self) -> None:
        from colnade.expr import Agg, ColumnRef

        bad_expr = Agg(source=ColumnRef(column=Users.id), agg_type="bad_agg")
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            _backend.translate_expr(bad_expr)

    def test_unsupported_function_call_raises(self) -> None:
        from colnade.expr import ColumnRef, FunctionCall

        bad_expr = FunctionCall(name="bad_function", args=(ColumnRef(column=Users.id),))
        with pytest.raises(ValueError, match="Unsupported FunctionCall"):
            _backend.translate_expr(bad_expr)

    def test_unsupported_expression_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported expression type"):
            _backend.translate_expr("not an expr")  # type: ignore[arg-type]

    def test_unsupported_list_op_raises(self) -> None:
        from colnade.expr import ColumnRef, ListOp

        bad_expr = ListOp(list_expr=ColumnRef(column=Users.id), op="bad_list_op", args=())
        with pytest.raises(ValueError, match="Unsupported ListOp"):
            _backend.translate_expr(bad_expr)


# ---------------------------------------------------------------------------
# Unary operations executed through the backend
# ---------------------------------------------------------------------------


class TestUnaryOpsExecution:
    def test_negation(self) -> None:
        df = _scores_ddf()
        result = df.with_columns((-Scores.score).alias(Scores.score))
        assert result._data.compute()["score"].tolist()[0] == -85.5

    def test_invert_boolean(self) -> None:
        df = _users_ddf()
        result = df.filter(~(Users.age > 30))
        names = set(result._data.compute()["name"].tolist())
        assert names == {"Alice", "Bob", "Diana"}

    def test_is_null(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", pd.NA], dtype=pd.StringDtype()),
                "age": pd.array([30, 25], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=NullableUsers, _backend=_backend)
        result = df.filter(NullableUsers.name.is_null())
        assert result._data.compute().shape[0] == 1

    def test_is_not_null(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", pd.NA], dtype=pd.StringDtype()),
                "age": pd.array([30, 25], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=NullableUsers, _backend=_backend)
        result = df.filter(NullableUsers.name.is_not_null())
        assert result._data.compute().shape[0] == 1

    def test_is_nan(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Bob"], dtype=pd.StringDtype()),
                "score": pd.array([85.5, float("nan")], dtype="Float64"),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Scores, _backend=_backend)
        result = df.filter(Scores.score.is_nan())
        assert result._data.compute().shape[0] == 1


# ---------------------------------------------------------------------------
# Function calls: assert_non_null, cast, over, dt_truncate
# ---------------------------------------------------------------------------


class TestFunctionCallExecution:
    def test_assert_non_null(self) -> None:
        df = _users_ddf()
        result = df.with_columns(Users.age.assert_non_null().alias(Users.age))
        assert result._data.compute()["age"].tolist() == [30, 25, 35, 28, 40]

    def test_cast(self) -> None:
        df = _users_ddf()
        result = df.with_columns(Users.age.cast(Float64).alias(Users.age))
        assert result._data.compute()["age"].dtype == pd.Float64Dtype()

    def test_over_window(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2, 3, 4], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Alice", "Bob", "Bob"], dtype=pd.StringDtype()),
                "age": pd.array([30, 25, 35, 28], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Users, _backend=_backend)
        result = df.with_columns(Users.age.over(Users.name).alias(Users.age))
        assert result._data.compute().shape[0] == 4

    def test_dt_truncate(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2], dtype=pd.UInt64Dtype()),
                "ts": pd.array(
                    [
                        pd.Timestamp("2025-03-15 10:30:45"),
                        pd.Timestamp("2025-06-20 14:15:30"),
                    ],
                    dtype=pd.ArrowDtype(pa.timestamp("us")),
                ),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Events, _backend=_backend)
        result = df.with_columns(Events.ts.dt_truncate("D").alias(Events.ts))
        assert result._data.compute()["ts"].tolist()[0] == pd.Timestamp("2025-03-15")


# ---------------------------------------------------------------------------
# Ungrouped aggregation
# ---------------------------------------------------------------------------


class TestUngroupedAgg:
    def test_ungrouped_single_agg(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
                "age": pd.array([30, 25, 35], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Users, _backend=_backend)
        result = df.agg(Users.age.sum().alias(Users.age))
        computed = result._data.compute()
        assert computed.shape[0] == 1
        assert computed["age"].iloc[0] == 90

    def test_ungrouped_multi_agg(self) -> None:
        data = pd.DataFrame(
            {
                "id": pd.array([1, 2, 3], dtype=pd.UInt64Dtype()),
                "name": pd.array(["Alice", "Bob", "Charlie"], dtype=pd.StringDtype()),
                "age": pd.array([30, 25, 35], dtype=pd.UInt64Dtype()),
            }
        )
        ddf = dd.from_pandas(data, npartitions=1)
        df = DataFrame(_data=ddf, _schema=Users, _backend=_backend)
        result = df.agg(
            Users.age.sum().alias(Users.age),
            Users.id.count().alias(Users.id),
        )
        computed = result._data.compute()
        assert computed.shape[0] == 1
        assert computed["age"].iloc[0] == 90
        assert computed["id"].iloc[0] == 3
