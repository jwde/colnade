"""Integration tests for Field() value-level constraint validation — Pandas backend."""

from __future__ import annotations

import pandas as pd
import pytest

import colnade.validation
from colnade import Column, DataFrame, Float64, Schema, SchemaError, UInt64, Utf8
from colnade.constraints import Field, schema_check
from colnade_pandas.adapter import PandasBackend


class Constrained(Schema):
    id: Column[UInt64] = Field(unique=True)
    age: Column[UInt64] = Field(ge=0, le=150)
    name: Column[Utf8] = Field(min_length=1)
    score: Column[Float64] = Field(ge=0.0, le=100.0)
    status: Column[Utf8] = Field(isin=["active", "inactive"])


def _make_df(overrides: dict | None = None) -> DataFrame[Constrained]:
    data = {
        "id": [1, 2, 3],
        "age": [25, 30, 45],
        "name": ["Alice", "Bob", "Carol"],
        "score": [85.0, 90.0, 75.0],
        "status": ["active", "inactive", "active"],
    }
    if overrides:
        data.update(overrides)
    pdf = pd.DataFrame(data)
    pdf = pdf.astype(
        {
            "id": pd.UInt64Dtype(),
            "age": pd.UInt64Dtype(),
            "name": pd.StringDtype(),
            "score": pd.Float64Dtype(),
            "status": pd.StringDtype(),
        }
    )
    backend = PandasBackend()
    return DataFrame(_data=pdf, _schema=Constrained, _backend=backend)


class TestValidDataPasses:
    def test_valid_data(self) -> None:
        df = _make_df()
        result = df.validate()
        assert result is df


class TestGeViolation:
    def test_ge_violation(self) -> None:
        df = _make_df({"score": [-1.0, 50.0, 101.0]})
        with pytest.raises(SchemaError, match="ge=0.0"):
            df.validate()


class TestLeViolation:
    def test_le_violation(self) -> None:
        df = _make_df({"score": [85.0, 90.0, 150.0]})
        with pytest.raises(SchemaError, match="le=100.0"):
            df.validate()


class TestGtViolation:
    def test_gt_violation(self) -> None:
        class GtSchema(Schema):
            val: Column[Float64] = Field(gt=0.0)

        pdf = pd.DataFrame({"val": [0.0, 1.0, 2.0]}).astype({"val": pd.Float64Dtype()})
        backend = PandasBackend()
        df: DataFrame[GtSchema] = DataFrame(_data=pdf, _schema=GtSchema, _backend=backend)
        with pytest.raises(SchemaError, match="gt=0.0"):
            df.validate()


class TestLtViolation:
    def test_lt_violation(self) -> None:
        class LtSchema(Schema):
            val: Column[Float64] = Field(lt=100.0)

        pdf = pd.DataFrame({"val": [50.0, 100.0]}).astype({"val": pd.Float64Dtype()})
        backend = PandasBackend()
        df: DataFrame[LtSchema] = DataFrame(_data=pdf, _schema=LtSchema, _backend=backend)
        with pytest.raises(SchemaError, match="lt=100.0"):
            df.validate()


class TestUniqueViolation:
    def test_unique_violation(self) -> None:
        df = _make_df({"id": [1, 1, 2]})
        with pytest.raises(SchemaError, match="unique"):
            df.validate()


class TestMinLengthViolation:
    def test_min_length_violation(self) -> None:
        df = _make_df({"name": ["Alice", "", "Carol"]})
        with pytest.raises(SchemaError, match="min_length=1"):
            df.validate()


class TestMaxLengthViolation:
    def test_max_length_violation(self) -> None:
        class MaxLen(Schema):
            name: Column[Utf8] = Field(max_length=3)

        pdf = pd.DataFrame({"name": ["AB", "ABCDEF"]}).astype({"name": pd.StringDtype()})
        backend = PandasBackend()
        df: DataFrame[MaxLen] = DataFrame(_data=pdf, _schema=MaxLen, _backend=backend)
        with pytest.raises(SchemaError, match="max_length=3"):
            df.validate()


class TestPatternViolation:
    def test_pattern_violation(self) -> None:
        class WithEmail(Schema):
            email: Column[Utf8] = Field(pattern=r"^[^@]+@[^@]+\.[^@]+$")

        pdf = pd.DataFrame({"email": ["a@b.com", "invalid"]}).astype({"email": pd.StringDtype()})
        backend = PandasBackend()
        df: DataFrame[WithEmail] = DataFrame(_data=pdf, _schema=WithEmail, _backend=backend)
        with pytest.raises(SchemaError, match="pattern="):
            df.validate()


class TestIsinViolation:
    def test_isin_violation(self) -> None:
        df = _make_df({"status": ["active", "deleted", "active"]})
        with pytest.raises(SchemaError, match="isin="):
            df.validate()


class TestMultipleViolations:
    def test_multiple_violations_collected(self) -> None:
        df = _make_df(
            {
                "id": [1, 1, 2],
                "name": ["Alice", "", "Carol"],
            }
        )
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        err = exc_info.value
        assert len(err.value_violations) >= 2


class TestAutoValidateStructuralSkipsValues:
    def test_structural_level_skips_value_checks(self) -> None:
        colnade.validation.set_validation("structural")
        try:
            pdf = pd.DataFrame(
                {
                    "id": [1, 1, 2],
                    "age": [25, 30, 45],
                    "name": ["Alice", "", "Carol"],
                    "score": [85.0, 90.0, 75.0],
                    "status": ["active", "inactive", "active"],
                }
            )
            pdf = pdf.astype(
                {
                    "id": pd.UInt64Dtype(),
                    "age": pd.UInt64Dtype(),
                    "name": pd.StringDtype(),
                    "score": pd.Float64Dtype(),
                    "status": pd.StringDtype(),
                }
            )
            backend = PandasBackend()
            df: DataFrame[Constrained] = DataFrame(_data=pdf, _schema=Constrained, _backend=backend)
            result = df.with_raw(lambda raw: raw)
            assert result._schema is Constrained
        finally:
            colnade.validation._validation_level = None


class TestAutoValidateFullChecksValues:
    def test_full_level_checks_values(self) -> None:
        colnade.validation.set_validation("full")
        try:
            pdf = pd.DataFrame(
                {
                    "id": [1, 1, 2],
                    "age": [25, 30, 45],
                    "name": ["Alice", "Bob", "Carol"],
                    "score": [85.0, 90.0, 75.0],
                    "status": ["active", "inactive", "active"],
                }
            )
            pdf = pdf.astype(
                {
                    "id": pd.UInt64Dtype(),
                    "age": pd.UInt64Dtype(),
                    "name": pd.StringDtype(),
                    "score": pd.Float64Dtype(),
                    "status": pd.StringDtype(),
                }
            )
            backend = PandasBackend()
            df: DataFrame[Constrained] = DataFrame(_data=pdf, _schema=Constrained, _backend=backend)
            with pytest.raises(SchemaError, match="unique"):
                df.with_raw(lambda raw: raw)
        finally:
            colnade.validation._validation_level = None


class TestEmptyDataFrame:
    def test_empty_df_passes(self) -> None:
        pdf = pd.DataFrame(
            {
                "id": pd.array([], dtype=pd.UInt64Dtype()),
                "age": pd.array([], dtype=pd.UInt64Dtype()),
                "name": pd.array([], dtype=pd.StringDtype()),
                "score": pd.array([], dtype=pd.Float64Dtype()),
                "status": pd.array([], dtype=pd.StringDtype()),
            }
        )
        backend = PandasBackend()
        df: DataFrame[Constrained] = DataFrame(_data=pdf, _schema=Constrained, _backend=backend)
        result = df.validate()
        assert result is df


class TestSchemaCheckPandas:
    def test_schema_check_passes(self) -> None:
        class Range(Schema):
            lo: Column[UInt64]
            hi: Column[UInt64]

            @schema_check
            def lo_le_hi(cls):
                return Range.lo <= Range.hi

        pdf = pd.DataFrame({"lo": [1, 2, 3], "hi": [10, 20, 30]}).astype(
            {"lo": pd.UInt64Dtype(), "hi": pd.UInt64Dtype()}
        )
        backend = PandasBackend()
        df: DataFrame[Range] = DataFrame(_data=pdf, _schema=Range, _backend=backend)
        result = df.validate()
        assert result is df

    def test_schema_check_violation(self) -> None:
        class Range(Schema):
            lo: Column[UInt64]
            hi: Column[UInt64]

            @schema_check
            def lo_le_hi(cls):
                return Range.lo <= Range.hi

        pdf = pd.DataFrame({"lo": [1, 20, 3], "hi": [10, 5, 30]}).astype(
            {"lo": pd.UInt64Dtype(), "hi": pd.UInt64Dtype()}
        )
        backend = PandasBackend()
        df: DataFrame[Range] = DataFrame(_data=pdf, _schema=Range, _backend=backend)
        with pytest.raises(SchemaError, match="schema_check:lo_le_hi"):
            df.validate()
