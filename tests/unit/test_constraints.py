"""Unit tests for colnade.constraints (FieldInfo, Field, schema_check, ValueViolation)."""

from __future__ import annotations

import pytest

from colnade.constraints import Field, FieldInfo, SchemaCheck, ValueViolation, schema_check


class TestFieldInfo:
    def test_default_values(self) -> None:
        info = FieldInfo()
        assert info.ge is None
        assert info.gt is None
        assert info.le is None
        assert info.lt is None
        assert info.min_length is None
        assert info.max_length is None
        assert info.pattern is None
        assert info.unique is False
        assert info.isin is None
        assert info.mapped_from is None

    def test_has_constraints_false(self) -> None:
        assert FieldInfo().has_constraints() is False

    def test_has_constraints_true_ge(self) -> None:
        assert FieldInfo(ge=0).has_constraints() is True

    def test_has_constraints_true_unique(self) -> None:
        assert FieldInfo(unique=True).has_constraints() is True

    def test_has_constraints_true_isin(self) -> None:
        assert FieldInfo(isin=["a", "b"]).has_constraints() is True

    def test_has_constraints_true_pattern(self) -> None:
        assert FieldInfo(pattern=r"^\d+$").has_constraints() is True

    def test_has_constraints_mapped_from_only_false(self) -> None:
        """mapped_from alone is not a value constraint."""
        assert FieldInfo(mapped_from="something").has_constraints() is False

    def test_ge_gt_mutual_exclusion(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both 'ge' and 'gt'"):
            FieldInfo(ge=0, gt=0)

    def test_le_lt_mutual_exclusion(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both 'le' and 'lt'"):
            FieldInfo(le=100, lt=100)

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            FieldInfo(pattern="[invalid")

    def test_valid_regex_ok(self) -> None:
        info = FieldInfo(pattern=r"^[^@]+@[^@]+\.[^@]+$")
        assert info.pattern is not None

    def test_frozen(self) -> None:
        info = FieldInfo(ge=0)
        with pytest.raises(AttributeError):
            info.ge = 10  # type: ignore[misc]

    def test_all_constraints(self) -> None:
        info = FieldInfo(
            ge=0,
            le=100,
            min_length=1,
            max_length=50,
            pattern=r"\w+",
            unique=True,
            isin=[1, 2],
        )
        assert info.has_constraints() is True


class TestField:
    def test_returns_fieldinfo(self) -> None:
        result = Field(ge=0, le=150)
        assert isinstance(result, FieldInfo)
        assert result.ge == 0
        assert result.le == 150

    def test_mapped_from_parameter(self) -> None:
        sentinel = object()
        result = Field(ge=0, mapped_from=sentinel)
        assert isinstance(result, FieldInfo)
        assert result.ge == 0
        assert result.mapped_from is sentinel

    def test_no_args(self) -> None:
        result = Field()
        assert isinstance(result, FieldInfo)
        assert result.has_constraints() is False


class TestSchemaCheck:
    def test_decorator_returns_schema_check(self) -> None:
        @schema_check
        def my_check(cls):
            return True

        assert isinstance(my_check, SchemaCheck)
        assert my_check.name == "my_check"

    def test_fn_is_stored(self) -> None:
        def my_fn(cls):
            return True

        check = schema_check(my_fn)
        assert check.fn is my_fn

    def test_repr(self) -> None:
        @schema_check
        def dates_ordered(cls):
            return True

        assert repr(dates_ordered) == "SchemaCheck('dates_ordered')"


class TestValueViolation:
    def test_construction(self) -> None:
        v = ValueViolation(column="age", constraint="ge=0", got_count=3, sample_values=[-1, -2, -3])
        assert v.column == "age"
        assert v.constraint == "ge=0"
        assert v.got_count == 3
        assert v.sample_values == [-1, -2, -3]

    def test_frozen(self) -> None:
        v = ValueViolation(column="age", constraint="ge=0", got_count=1, sample_values=[-1])
        with pytest.raises(AttributeError):
            v.column = "name"  # type: ignore[misc]
