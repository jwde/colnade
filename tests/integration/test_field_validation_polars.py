"""Integration tests for Field() value-level constraint validation — Polars backend."""

from __future__ import annotations

import polars as pl
import pytest

import colnade.validation
from colnade import Column, DataFrame, Float64, Schema, SchemaError, UInt64, Utf8
from colnade.constraints import Field, schema_check
from colnade_polars.adapter import PolarsBackend


class Constrained(Schema):
    id: Column[UInt64] = Field(unique=True)
    age: Column[UInt64] = Field(ge=0, le=150)
    name: Column[Utf8] = Field(min_length=1)
    email: Column[Utf8] = Field(pattern=r"^[^@]+@[^@]+\.[^@]+$")
    score: Column[Float64] = Field(ge=0.0, le=100.0)
    status: Column[Utf8] = Field(isin=["active", "inactive"])


def _make_df(overrides: dict | None = None) -> DataFrame[Constrained]:
    data = {
        "id": pl.Series([1, 2, 3], dtype=pl.UInt64),
        "age": pl.Series([25, 30, 45], dtype=pl.UInt64),
        "name": pl.Series(["Alice", "Bob", "Carol"], dtype=pl.Utf8),
        "email": pl.Series(["a@b.com", "c@d.com", "e@f.com"], dtype=pl.Utf8),
        "score": pl.Series([85.0, 90.0, 75.0], dtype=pl.Float64),
        "status": pl.Series(["active", "inactive", "active"], dtype=pl.Utf8),
    }
    if overrides:
        data.update(overrides)
    backend = PolarsBackend()
    return DataFrame(_data=pl.DataFrame(data), _schema=Constrained, _backend=backend)


class TestValidDataPasses:
    def test_valid_data(self) -> None:
        df = _make_df()
        result = df.validate()
        assert result is df


class TestGeViolation:
    def test_ge_violation(self) -> None:
        """UInt64 can't go negative, so test le instead with a low bound."""
        # Use score which is Float64
        df = _make_df({"score": pl.Series([-1.0, 50.0, 101.0], dtype=pl.Float64)})
        with pytest.raises(SchemaError, match="ge=0.0"):
            df.validate()


class TestLeViolation:
    def test_le_violation(self) -> None:
        df = _make_df({"score": pl.Series([85.0, 90.0, 150.0], dtype=pl.Float64)})
        with pytest.raises(SchemaError, match="le=100.0"):
            df.validate()


class TestGtLtViolation:
    def test_gt_violation(self) -> None:
        class GtSchema(Schema):
            val: Column[Float64] = Field(gt=0.0)

        data = pl.DataFrame({"val": pl.Series([0.0, 1.0, 2.0], dtype=pl.Float64)})
        backend = PolarsBackend()
        df: DataFrame[GtSchema] = DataFrame(_data=data, _schema=GtSchema, _backend=backend)
        with pytest.raises(SchemaError, match="gt=0.0"):
            df.validate()

    def test_lt_violation(self) -> None:
        class LtSchema(Schema):
            val: Column[Float64] = Field(lt=100.0)

        data = pl.DataFrame({"val": pl.Series([50.0, 100.0], dtype=pl.Float64)})
        backend = PolarsBackend()
        df: DataFrame[LtSchema] = DataFrame(_data=data, _schema=LtSchema, _backend=backend)
        with pytest.raises(SchemaError, match="lt=100.0"):
            df.validate()


class TestUniqueViolation:
    def test_unique_violation(self) -> None:
        df = _make_df({"id": pl.Series([1, 1, 2], dtype=pl.UInt64)})
        with pytest.raises(SchemaError, match="unique"):
            df.validate()


class TestMinLengthViolation:
    def test_min_length_violation(self) -> None:
        df = _make_df({"name": pl.Series(["Alice", "", "Carol"], dtype=pl.Utf8)})
        with pytest.raises(SchemaError, match="min_length=1"):
            df.validate()


class TestMaxLengthViolation:
    def test_max_length_violation(self) -> None:
        class MaxLen(Schema):
            name: Column[Utf8] = Field(max_length=3)

        data = pl.DataFrame({"name": pl.Series(["AB", "ABCDEF"], dtype=pl.Utf8)})
        backend = PolarsBackend()
        df: DataFrame[MaxLen] = DataFrame(_data=data, _schema=MaxLen, _backend=backend)
        with pytest.raises(SchemaError, match="max_length=3"):
            df.validate()


class TestPatternViolation:
    def test_pattern_violation(self) -> None:
        df = _make_df({"email": pl.Series(["a@b.com", "invalid", "e@f.com"], dtype=pl.Utf8)})
        with pytest.raises(SchemaError, match="pattern="):
            df.validate()


class TestIsinViolation:
    def test_isin_violation(self) -> None:
        df = _make_df({"status": pl.Series(["active", "deleted", "active"], dtype=pl.Utf8)})
        with pytest.raises(SchemaError, match="isin="):
            df.validate()


class TestMultipleViolations:
    def test_multiple_violations_collected(self) -> None:
        df = _make_df(
            {
                "id": pl.Series([1, 1, 2], dtype=pl.UInt64),
                "name": pl.Series(["Alice", "", "Carol"], dtype=pl.Utf8),
            }
        )
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        err = exc_info.value
        assert len(err.value_violations) >= 2


class TestSampleValues:
    def test_sample_values_populated(self) -> None:
        df = _make_df({"name": pl.Series(["Alice", "", ""], dtype=pl.Utf8)})
        with pytest.raises(SchemaError) as exc_info:
            df.validate()
        err = exc_info.value
        viol = [v for v in err.value_violations if v.column == "name"]
        assert len(viol) == 1
        assert "" in viol[0].sample_values


class TestExplicitValidateAlwaysChecks:
    def test_validate_checks_values_even_when_off(self) -> None:
        colnade.validation.set_validation("off")
        try:
            df = _make_df({"name": pl.Series(["Alice", "", "Carol"], dtype=pl.Utf8)})
            with pytest.raises(SchemaError, match="min_length=1"):
                df.validate()
        finally:
            colnade.validation._validation_level = None


class TestAutoValidateStructuralSkipsValues:
    def test_structural_level_skips_value_checks(self) -> None:
        colnade.validation.set_validation("structural")
        try:
            backend = PolarsBackend()
            bad_data = pl.DataFrame(
                {
                    "id": pl.Series([1, 1, 2], dtype=pl.UInt64),
                    "age": pl.Series([25, 30, 45], dtype=pl.UInt64),
                    "name": pl.Series(["Alice", "", "Carol"], dtype=pl.Utf8),
                    "email": pl.Series(["a@b.com", "invalid", "e@f.com"], dtype=pl.Utf8),
                    "score": pl.Series([85.0, 90.0, 75.0], dtype=pl.Float64),
                    "status": pl.Series(["active", "inactive", "active"], dtype=pl.Utf8),
                }
            )
            df: DataFrame[Constrained] = DataFrame(
                _data=bad_data, _schema=Constrained, _backend=backend
            )
            # with_raw auto-validates at "structural" level — should NOT raise on value constraints
            result = df.with_raw(lambda raw: raw)
            assert result._schema is Constrained
        finally:
            colnade.validation._validation_level = None


class TestAutoValidateFullChecksValues:
    def test_full_level_checks_values(self) -> None:
        colnade.validation.set_validation("full")
        try:
            backend = PolarsBackend()
            bad_data = pl.DataFrame(
                {
                    "id": pl.Series([1, 1, 2], dtype=pl.UInt64),
                    "age": pl.Series([25, 30, 45], dtype=pl.UInt64),
                    "name": pl.Series(["Alice", "Bob", "Carol"], dtype=pl.Utf8),
                    "email": pl.Series(["a@b.com", "c@d.com", "e@f.com"], dtype=pl.Utf8),
                    "score": pl.Series([85.0, 90.0, 75.0], dtype=pl.Float64),
                    "status": pl.Series(["active", "inactive", "active"], dtype=pl.Utf8),
                }
            )
            df: DataFrame[Constrained] = DataFrame(
                _data=bad_data, _schema=Constrained, _backend=backend
            )
            with pytest.raises(SchemaError, match="unique"):
                df.with_raw(lambda raw: raw)
        finally:
            colnade.validation._validation_level = None


class TestEmptyDataFrame:
    def test_empty_df_passes(self) -> None:
        data = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.UInt64),
                "age": pl.Series([], dtype=pl.UInt64),
                "name": pl.Series([], dtype=pl.Utf8),
                "email": pl.Series([], dtype=pl.Utf8),
                "score": pl.Series([], dtype=pl.Float64),
                "status": pl.Series([], dtype=pl.Utf8),
            }
        )
        backend = PolarsBackend()
        df: DataFrame[Constrained] = DataFrame(_data=data, _schema=Constrained, _backend=backend)
        result = df.validate()
        assert result is df


class TestSchemaCheckPolars:
    def test_schema_check_passes(self) -> None:
        class Range(Schema):
            lo: Column[UInt64]
            hi: Column[UInt64]

            @schema_check
            def lo_le_hi(cls):
                return Range.lo <= Range.hi

        data = pl.DataFrame(
            {
                "lo": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "hi": pl.Series([10, 20, 30], dtype=pl.UInt64),
            }
        )
        backend = PolarsBackend()
        df: DataFrame[Range] = DataFrame(_data=data, _schema=Range, _backend=backend)
        result = df.validate()
        assert result is df

    def test_schema_check_violation(self) -> None:
        class Range(Schema):
            lo: Column[UInt64]
            hi: Column[UInt64]

            @schema_check
            def lo_le_hi(cls):
                return Range.lo <= Range.hi

        data = pl.DataFrame(
            {
                "lo": pl.Series([1, 20, 3], dtype=pl.UInt64),
                "hi": pl.Series([10, 5, 30], dtype=pl.UInt64),
            }
        )
        backend = PolarsBackend()
        df: DataFrame[Range] = DataFrame(_data=data, _schema=Range, _backend=backend)
        with pytest.raises(SchemaError, match="schema_check:lo_le_hi"):
            df.validate()


class TestNoConstraintsNoOp:
    def test_schema_without_constraints(self) -> None:
        class Plain(Schema):
            name: Column[Utf8]

        data = pl.DataFrame({"name": pl.Series(["Alice"], dtype=pl.Utf8)})
        backend = PolarsBackend()
        df: DataFrame[Plain] = DataFrame(_data=data, _schema=Plain, _backend=backend)
        result = df.validate()
        assert result is df
