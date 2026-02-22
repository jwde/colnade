"""Integration tests for typed DataFrame construction (from_rows, from_dict)."""

from __future__ import annotations

import pytest

import colnade
from colnade import Column, DataFrame, Float64, Schema, UInt64, Utf8, ValidationLevel
from colnade.constraints import Field

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class UsersConstrained(Schema):
    id: Column[UInt64] = Field(unique=True)
    name: Column[Utf8] = Field(min_length=1)
    age: Column[UInt64] = Field(ge=0, le=150)
    score: Column[Float64] = Field(ge=0.0, le=100.0)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DICT: dict[str, list] = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35],
    "score": [85.0, 92.5, 78.0],
}


def _sample_rows() -> list:
    return [
        Users.Row(id=1, name="Alice", age=30, score=85.0),
        Users.Row(id=2, name="Bob", age=25, score=92.5),
        Users.Row(id=3, name="Charlie", age=35, score=78.0),
    ]


def _sample_dicts() -> list[dict]:
    return [
        {"id": 1, "name": "Alice", "age": 30, "score": 85.0},
        {"id": 2, "name": "Bob", "age": 25, "score": 92.5},
        {"id": 3, "name": "Charlie", "age": 35, "score": 78.0},
    ]


# ===========================================================================
# Polars
# ===========================================================================


class TestPolarsFromDict:
    def test_basic(self) -> None:
        from colnade_polars import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        assert isinstance(df, DataFrame)
        assert df.height == 3
        assert df.width == 4

    def test_dtype_coercion(self) -> None:
        """Plain Python ints are coerced to UInt64, floats to Float64."""
        from colnade_polars import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        native = df.to_native()
        import polars as pl

        assert native["id"].dtype == pl.UInt64
        assert native["name"].dtype == pl.String
        assert native["age"].dtype == pl.UInt64
        assert native["score"].dtype == pl.Float64

    def test_values_preserved(self) -> None:
        from colnade_polars import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        rows = list(df.iter_rows_as(dict))
        assert rows[0] == {"id": 1, "name": "Alice", "age": 30, "score": 85.0}
        assert rows[2] == {"id": 3, "name": "Charlie", "age": 35, "score": 78.0}

    def test_empty(self) -> None:
        from colnade_polars import from_dict

        empty: dict[str, list] = {"id": [], "name": [], "age": [], "score": []}
        df = from_dict(Users, empty)
        assert df.height == 0
        assert df.width == 4

    def test_operations_work(self) -> None:
        """DataFrame from from_dict supports normal operations."""
        from colnade_polars import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        filtered = df.filter(Users.age > 28)
        assert filtered.height == 2  # Alice (30), Charlie (35)


class TestPolarsFromRows:
    def test_from_schema_row(self) -> None:
        from colnade_polars import from_rows

        df = from_rows(Users, _sample_rows())
        assert df.height == 3
        assert df.width == 4

    def test_from_dicts(self) -> None:
        from colnade_polars import from_rows

        df = from_rows(Users, _sample_dicts())
        assert df.height == 3
        rows = list(df.iter_rows_as(dict))
        assert rows[1]["name"] == "Bob"

    def test_roundtrip(self) -> None:
        """from_rows → iter_rows_as roundtrip preserves data."""
        from colnade_polars import from_rows

        original = _sample_rows()
        df = from_rows(Users, original)
        roundtripped = list(df.iter_rows_as(Users.Row))
        for orig, rt in zip(original, roundtripped, strict=True):
            assert orig.id == rt.id
            assert orig.name == rt.name
            assert orig.age == rt.age
            assert orig.score == rt.score

    def test_empty(self) -> None:
        from colnade_polars import from_rows

        df = from_rows(Users, [])
        assert df.height == 0


# ===========================================================================
# Pandas
# ===========================================================================


class TestPandasFromDict:
    def test_basic(self) -> None:
        from colnade_pandas import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        assert isinstance(df, DataFrame)
        assert df.height == 3
        assert df.width == 4

    def test_dtype_coercion(self) -> None:
        import pandas as pd

        from colnade_pandas import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        native = df.to_native()
        assert native["id"].dtype == pd.UInt64Dtype()
        assert native["name"].dtype == pd.StringDtype()
        assert native["age"].dtype == pd.UInt64Dtype()
        assert native["score"].dtype == pd.Float64Dtype()

    def test_values_preserved(self) -> None:
        from colnade_pandas import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        rows = list(df.iter_rows_as(dict))
        assert rows[0]["name"] == "Alice"
        assert rows[2]["age"] == 35

    def test_operations_work(self) -> None:
        from colnade_pandas import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        filtered = df.filter(Users.age > 28)
        assert filtered.height == 2


class TestPandasFromRows:
    def test_from_schema_row(self) -> None:
        from colnade_pandas import from_rows

        df = from_rows(Users, _sample_rows())
        assert df.height == 3
        assert df.width == 4

    def test_from_dicts(self) -> None:
        from colnade_pandas import from_rows

        df = from_rows(Users, _sample_dicts())
        assert df.height == 3


# ===========================================================================
# Dask
# ===========================================================================


class TestDaskFromDict:
    def test_basic(self) -> None:
        from colnade_dask import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        assert isinstance(df, DataFrame)
        assert df.height == 3
        assert df.width == 4

    def test_values_preserved(self) -> None:
        from colnade_dask import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        rows = list(df.iter_rows_as(dict))
        assert rows[0]["name"] == "Alice"

    def test_operations_work(self) -> None:
        from colnade_dask import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        filtered = df.filter(Users.age > 28)
        assert filtered.height == 2


class TestDaskFromRows:
    def test_from_schema_row(self) -> None:
        from colnade_dask import from_rows

        df = from_rows(Users, _sample_rows())
        assert df.height == 3

    def test_from_dicts(self) -> None:
        from colnade_dask import from_rows

        df = from_rows(Users, _sample_dicts())
        assert df.height == 3


# ===========================================================================
# Validation integration
# ===========================================================================


class TestValidation:
    def setup_method(self) -> None:
        colnade.set_validation(ValidationLevel.OFF)

    def teardown_method(self) -> None:
        colnade.set_validation(ValidationLevel.OFF)

    def test_structural_validation_on_construction(self) -> None:
        """STRUCTURAL validation catches dtype issues at construction."""
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.STRUCTURAL)
        # Valid data should pass
        df = from_dict(Users, SAMPLE_DICT)
        assert df.height == 3

    def test_full_validation_checks_constraints(self) -> None:
        """FULL validation checks Field() constraints at construction."""
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.FULL)
        # Valid data
        df = from_dict(
            UsersConstrained,
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
                "score": [85.0, 92.5, 78.0],
            },
        )
        assert df.height == 3

    def test_full_validation_rejects_bad_values(self) -> None:
        """FULL validation rejects data that violates Field() constraints."""
        from colnade import SchemaError
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.FULL)
        with pytest.raises(SchemaError):
            from_dict(
                UsersConstrained,
                {
                    "id": [1, 2, 3],
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [30, 25, 200],  # violates le=150
                    "score": [85.0, 92.5, 78.0],
                },
            )

    def test_off_skips_validation(self) -> None:
        """OFF validation does not check anything."""
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.OFF)
        # Would fail FULL validation (age=200) but passes with OFF
        df = from_dict(
            UsersConstrained,
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 200],
                "score": [85.0, 92.5, 78.0],
            },
        )
        assert df.height == 3


# ===========================================================================
# rows_to_dict unit tests
# ===========================================================================


class TestRowsToDict:
    def test_from_row_objects(self) -> None:
        from colnade.dataframe import rows_to_dict

        rows = _sample_rows()
        result = rows_to_dict(rows, Users)
        assert result["id"] == [1, 2, 3]
        assert result["name"] == ["Alice", "Bob", "Charlie"]
        assert result["age"] == [30, 25, 35]
        assert result["score"] == [85.0, 92.5, 78.0]

    def test_from_dicts(self) -> None:
        from colnade.dataframe import rows_to_dict

        result = rows_to_dict(_sample_dicts(), Users)
        assert result["id"] == [1, 2, 3]
        assert result["name"] == ["Alice", "Bob", "Charlie"]

    def test_empty(self) -> None:
        from colnade.dataframe import rows_to_dict

        result = rows_to_dict([], Users)
        assert result == {"id": [], "name": [], "age": [], "score": []}
