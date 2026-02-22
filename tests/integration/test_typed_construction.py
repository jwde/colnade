"""Integration tests for typed DataFrame construction (from_rows, from_dict)."""

from __future__ import annotations

import pytest

import colnade
from colnade import Column, DataFrame, Float64, Row, Schema, UInt64, Utf8, ValidationLevel
from colnade.constraints import Field

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class Orders(Schema):
    id: Column[UInt64]
    amount: Column[Float64]


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


# ===========================================================================
# Row[S] base class
# ===========================================================================


class TestRowType:
    def test_row_is_instance_of_row_base(self) -> None:
        row = Users.Row(id=1, name="Alice", age=30, score=85.0)
        assert isinstance(row, Row)

    def test_different_schemas_produce_different_row_types(self) -> None:
        user_row = Users.Row(id=1, name="Alice", age=30, score=85.0)
        order_row = Orders.Row(id=1, amount=99.0)
        # Both are Row instances
        assert isinstance(user_row, Row)
        assert isinstance(order_row, Row)
        # But different classes
        assert type(user_row) is not type(order_row)


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
        import polars as pl

        from colnade_polars import from_dict

        df = from_dict(Users, SAMPLE_DICT)
        native = df.to_native()
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

    def test_empty(self) -> None:
        from colnade.dataframe import rows_to_dict

        result = rows_to_dict([], Users)
        assert result == {"id": [], "name": [], "age": [], "score": []}

    def test_mismatched_row_type_raises(self) -> None:
        """rows_to_dict raises KeyError when row doesn't match schema columns."""
        from colnade.dataframe import rows_to_dict

        orders_rows = [Orders.Row(id=1, amount=99.0)]
        with pytest.raises(KeyError):
            rows_to_dict(orders_rows, Users)


# ===========================================================================
# Row construction errors
# ===========================================================================


class TestRowConstructionErrors:
    def test_missing_fields_raises(self) -> None:
        """Row construction fails when required fields are omitted."""
        with pytest.raises(TypeError, match="missing.*required"):
            Users.Row(id=1, name="Alice")  # missing age, score

    def test_extra_field_raises(self) -> None:
        """Row construction fails on unexpected keyword arguments."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Users.Row(id=1, name="Alice", age=30, score=85.0, extra="x")

    def test_frozen_row_cannot_be_mutated(self) -> None:
        """Row instances are frozen dataclasses — assignment raises."""
        row = Users.Row(id=1, name="Alice", age=30, score=85.0)
        with pytest.raises(AttributeError):
            row.name = "Bob"  # type: ignore[misc]


# ===========================================================================
# from_dict negative cases — Polars
# ===========================================================================


class TestPolarsFromDictErrors:
    def test_missing_column_raises(self) -> None:
        """from_dict raises when a schema column is missing from the dict."""
        from colnade_polars import from_dict

        with pytest.raises((KeyError, ValueError)):
            from_dict(Users, {"id": [1], "name": ["a"], "age": [30]})

    def test_extra_column_raises(self) -> None:
        """Polars rejects data columns not present in the schema."""
        from colnade_polars import from_dict

        with pytest.raises(ValueError):
            from_dict(
                Users,
                {**SAMPLE_DICT, "extra": ["x", "y", "z"]},
            )

    def test_incompatible_dtype_raises(self) -> None:
        """Strings cannot be coerced to UInt64."""
        from colnade_polars import from_dict

        with pytest.raises(TypeError):
            from_dict(
                Users,
                {
                    "id": ["not", "a", "number"],
                    "name": ["a", "b", "c"],
                    "age": [1, 2, 3],
                    "score": [1.0, 2.0, 3.0],
                },
            )

    def test_ragged_column_lengths_raises(self) -> None:
        """Columns with different lengths are rejected."""
        from polars.exceptions import ShapeError

        from colnade_polars import from_dict

        with pytest.raises(ShapeError):
            from_dict(
                Users,
                {
                    "id": [1, 2, 3],
                    "name": ["a", "b"],  # only 2
                    "age": [1, 2, 3],
                    "score": [1.0, 2.0, 3.0],
                },
            )

    def test_empty_dict_creates_empty_frame(self) -> None:
        """Polars uses schema to create a 0-row frame from an empty dict."""
        from colnade_polars import from_dict

        df = from_dict(Users, {})
        assert df.height == 0
        assert df.width == 4


# ===========================================================================
# from_dict negative cases — Pandas
# ===========================================================================


class TestPandasFromDictErrors:
    def test_missing_column_raises(self) -> None:
        from colnade_pandas import from_dict

        with pytest.raises(KeyError):
            from_dict(Users, {"id": [1], "name": ["a"], "age": [30]})

    def test_incompatible_dtype_raises(self) -> None:
        from colnade_pandas import from_dict

        with pytest.raises((TypeError, ValueError)):
            from_dict(
                Users,
                {
                    "id": ["not", "a", "number"],
                    "name": ["a", "b", "c"],
                    "age": [1, 2, 3],
                    "score": [1.0, 2.0, 3.0],
                },
            )

    def test_ragged_column_lengths_raises(self) -> None:
        from colnade_pandas import from_dict

        with pytest.raises(ValueError):
            from_dict(
                Users,
                {
                    "id": [1, 2, 3],
                    "name": ["a", "b"],
                    "age": [1, 2, 3],
                    "score": [1.0, 2.0, 3.0],
                },
            )

    def test_empty_dict_raises(self) -> None:
        from colnade_pandas import from_dict

        with pytest.raises((KeyError, ValueError)):
            from_dict(Users, {})


# ===========================================================================
# from_dict negative cases — Dask
# ===========================================================================


class TestDaskFromDictErrors:
    def test_missing_column_raises(self) -> None:
        from colnade_dask import from_dict

        with pytest.raises(KeyError):
            from_dict(Users, {"id": [1], "name": ["a"], "age": [30]})

    def test_incompatible_dtype_raises(self) -> None:
        from colnade_dask import from_dict

        with pytest.raises((TypeError, ValueError)):
            from_dict(
                Users,
                {
                    "id": ["not", "a", "number"],
                    "name": ["a", "b", "c"],
                    "age": [1, 2, 3],
                    "score": [1.0, 2.0, 3.0],
                },
            )

    def test_ragged_column_lengths_raises(self) -> None:
        from colnade_dask import from_dict

        with pytest.raises(ValueError):
            from_dict(
                Users,
                {
                    "id": [1, 2, 3],
                    "name": ["a", "b"],
                    "age": [1, 2, 3],
                    "score": [1.0, 2.0, 3.0],
                },
            )

    def test_empty_dict_raises(self) -> None:
        from colnade_dask import from_dict

        with pytest.raises((KeyError, ValueError)):
            from_dict(Users, {})


# ===========================================================================
# from_rows negative cases
# ===========================================================================


class TestFromRowsErrors:
    def test_mismatched_row_type_polars(self) -> None:
        """Passing Orders.Row to from_rows(Users, ...) raises KeyError."""
        from colnade_polars import from_rows

        with pytest.raises(KeyError):
            from_rows(Users, [Orders.Row(id=1, amount=99.0)])

    def test_mismatched_row_type_pandas(self) -> None:
        from colnade_pandas import from_rows

        with pytest.raises(KeyError):
            from_rows(Users, [Orders.Row(id=1, amount=99.0)])

    def test_mismatched_row_type_dask(self) -> None:
        from colnade_dask import from_rows

        with pytest.raises(KeyError):
            from_rows(Users, [Orders.Row(id=1, amount=99.0)])


# ===========================================================================
# Validation negative cases
# ===========================================================================


class TestValidationNegative:
    def setup_method(self) -> None:
        colnade.set_validation(ValidationLevel.OFF)

    def teardown_method(self) -> None:
        colnade.set_validation(ValidationLevel.OFF)

    def test_structural_rejects_null_in_non_nullable(self) -> None:
        """STRUCTURAL catches nulls in non-nullable columns at construction."""
        from colnade import SchemaError
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.STRUCTURAL)
        with pytest.raises(SchemaError):
            from_dict(
                Users,
                {
                    "id": [1, None, 3],
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [30, 25, 35],
                    "score": [85.0, 92.5, 78.0],
                },
            )

    def test_structural_does_not_check_value_constraints(self) -> None:
        """STRUCTURAL allows values that violate Field() constraints."""
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.STRUCTURAL)
        df = from_dict(
            UsersConstrained,
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 200],  # violates le=150
                "score": [85.0, 92.5, 78.0],
            },
        )
        assert df.height == 3

    def test_full_rejects_multiple_constraint_violations(self) -> None:
        """FULL validation catches violations across multiple columns."""
        from colnade import SchemaError
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.FULL)
        with pytest.raises(SchemaError):
            from_dict(
                UsersConstrained,
                {
                    "id": [1, 1, 3],  # violates unique
                    "name": ["Alice", "", "Charlie"],  # violates min_length=1
                    "age": [30, 25, 200],  # violates le=150
                    "score": [85.0, 92.5, 78.0],
                },
            )

    def test_explicit_validate_catches_nulls_after_off(self) -> None:
        """df.validate() catches errors even when global level is OFF."""
        from colnade import SchemaError
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.OFF)
        df = from_dict(
            Users,
            {
                "id": [1, None, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
                "score": [85.0, 92.5, 78.0],
            },
        )
        assert df.height == 3  # construction succeeds

        with pytest.raises(SchemaError):
            df.validate()

    def test_explicit_validate_catches_constraint_violations_after_off(self) -> None:
        """df.validate() runs FULL checks regardless of global level."""
        from colnade import SchemaError
        from colnade_polars import from_dict

        colnade.set_validation(ValidationLevel.OFF)
        df = from_dict(
            UsersConstrained,
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 200],  # violates le=150
                "score": [85.0, 92.5, 78.0],
            },
        )
        assert df.height == 3

        with pytest.raises(SchemaError):
            df.validate()
