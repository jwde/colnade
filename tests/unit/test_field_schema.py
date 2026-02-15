"""Unit tests for SchemaMeta integration with Field() and @schema_check."""

from __future__ import annotations

from colnade import Column, Schema, UInt64, Utf8
from colnade.constraints import Field, FieldInfo, SchemaCheck, schema_check


class TestFieldOnSchema:
    def test_field_stores_field_info(self) -> None:
        class FieldAge(Schema):
            age: Column[UInt64] = Field(ge=0, le=150)

        assert isinstance(FieldAge.age._field_info, FieldInfo)
        assert FieldAge.age._field_info.ge == 0
        assert FieldAge.age._field_info.le == 150

    def test_field_unique(self) -> None:
        class FieldUnique(Schema):
            id: Column[UInt64] = Field(unique=True)

        assert FieldUnique.id._field_info is not None
        assert FieldUnique.id._field_info.unique is True

    def test_field_pattern(self) -> None:
        class FieldPattern(Schema):
            email: Column[Utf8] = Field(pattern=r"^[^@]+@[^@]+\.[^@]+$")

        assert FieldPattern.email._field_info is not None
        assert FieldPattern.email._field_info.pattern is not None

    def test_field_with_mapped_from(self) -> None:
        class FieldSrc(Schema):
            age: Column[UInt64]

        class FieldTgt(Schema):
            age: Column[UInt64] = Field(ge=0, mapped_from=FieldSrc.age)

        assert FieldTgt.age._field_info is not None
        assert FieldTgt.age._field_info.ge == 0
        assert FieldTgt.age._mapped_from is FieldSrc.age

    def test_no_field_has_none(self) -> None:
        class PlainCol(Schema):
            name: Column[Utf8]

        assert PlainCol.name._field_info is None

    def test_plain_mapped_from_still_works(self) -> None:
        from colnade import mapped_from

        class MapSrc(Schema):
            id: Column[UInt64]

        class MapTgt(Schema):
            source_id: Column[UInt64] = mapped_from(MapSrc.id)

        assert MapTgt.source_id._mapped_from is MapSrc.id
        assert MapTgt.source_id._field_info is None

    def test_mixed_field_and_plain(self) -> None:
        class MixedCols(Schema):
            id: Column[UInt64] = Field(unique=True)
            name: Column[Utf8]

        assert MixedCols.id._field_info is not None
        assert MixedCols.id._field_info.unique is True
        assert MixedCols.name._field_info is None


class TestSchemaCheckOnSchema:
    def test_schema_check_collected(self) -> None:
        class CheckEvents(Schema):
            start: Column[UInt64]
            end: Column[UInt64]

            @schema_check
            def start_before_end(cls):
                return CheckEvents.start <= CheckEvents.end

        assert len(CheckEvents._schema_checks) == 1
        assert isinstance(CheckEvents._schema_checks[0], SchemaCheck)
        assert CheckEvents._schema_checks[0].name == "start_before_end"

    def test_no_checks_empty_list(self) -> None:
        class NoChecks(Schema):
            name: Column[Utf8]

        assert _has_no_schema_checks(NoChecks)


def _has_no_schema_checks(schema: type) -> bool:
    return getattr(schema, "_schema_checks", []) == []


class TestFieldInheritance:
    def test_subclass_inherits_field_info(self) -> None:
        class InhBase(Schema):
            age: Column[UInt64] = Field(ge=0, le=150)

        class InhChild(InhBase):
            email: Column[Utf8]

        assert InhChild.age._field_info is not None
        assert InhChild.age._field_info.ge == 0
        assert InhChild.age._field_info.le == 150
        assert InhChild.email._field_info is None

    def test_subclass_can_override_field_info(self) -> None:
        class OverBase(Schema):
            age: Column[UInt64] = Field(ge=0, le=150)

        class OverChild(OverBase):
            age: Column[UInt64] = Field(ge=18, le=120)

        assert OverChild.age._field_info is not None
        assert OverChild.age._field_info.ge == 18
        assert OverChild.age._field_info.le == 120
        # Parent unchanged
        assert OverBase.age._field_info.ge == 0

    def test_subclass_inherits_schema_checks(self) -> None:
        class ChkBase(Schema):
            lo: Column[UInt64]
            hi: Column[UInt64]

            @schema_check
            def lo_le_hi(cls):
                return ChkBase.lo <= ChkBase.hi

        class ChkChild(ChkBase):
            mid: Column[UInt64]

        assert len(ChkChild._schema_checks) == 1
        assert ChkChild._schema_checks[0].name == "lo_le_hi"

    def test_subclass_extends_schema_checks(self) -> None:
        class ExtBase(Schema):
            lo: Column[UInt64]
            hi: Column[UInt64]

            @schema_check
            def lo_le_hi(cls):
                return ExtBase.lo <= ExtBase.hi

        class ExtChild(ExtBase):
            mid: Column[UInt64]

            @schema_check
            def mid_in_range(cls):
                return (ExtChild.lo <= ExtChild.mid) & (ExtChild.mid <= ExtChild.hi)

        assert len(ExtChild._schema_checks) == 2
        names = [c.name for c in ExtChild._schema_checks]
        assert "lo_le_hi" in names
        assert "mid_in_range" in names


class TestSchemaErrorValueViolations:
    def test_schema_error_with_value_violations(self) -> None:
        from colnade import SchemaError
        from colnade.constraints import ValueViolation

        v = ValueViolation(column="age", constraint="ge=0", got_count=2, sample_values=[-1, -5])
        err = SchemaError(value_violations=[v])
        assert "Value violations" in str(err)
        assert "age" in str(err)
        assert "ge=0" in str(err)

    def test_schema_error_backward_compat(self) -> None:
        from colnade import SchemaError

        err = SchemaError(missing_columns=["foo"])
        assert err.value_violations == []
        assert "Missing columns" in str(err)
