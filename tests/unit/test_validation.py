"""Unit tests for the runtime validation toggle and literal type checking."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from colnade import Column, Schema, UInt64, Utf8
from colnade.validation import (
    ValidationLevel,
    check_literal_type,
    get_validation_level,
    is_validation_enabled,
    set_validation,
)


class TestValidationToggle:
    def teardown_method(self) -> None:
        # Reset global state after each test
        set_validation.__wrapped__ if hasattr(set_validation, "__wrapped__") else None
        import colnade.validation

        colnade.validation._validation_level = None

    def test_default_is_disabled(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {}, clear=True):
            assert is_validation_enabled() is False

    def test_set_validation_true(self) -> None:
        set_validation(True)
        assert is_validation_enabled() is True

    def test_set_validation_false(self) -> None:
        set_validation(False)
        assert is_validation_enabled() is False

    def test_env_var_1(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "1"}):
            assert is_validation_enabled() is True

    def test_env_var_true(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "true"}):
            assert is_validation_enabled() is True

    def test_env_var_yes(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "yes"}):
            assert is_validation_enabled() is True

    def test_env_var_TRUE_case_insensitive(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "TRUE"}):
            assert is_validation_enabled() is True

    def test_env_var_invalid(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "nope"}):
            assert is_validation_enabled() is False

    def test_set_overrides_env_var(self) -> None:
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "1"}):
            set_validation(False)
            assert is_validation_enabled() is False

    def test_exports(self) -> None:
        import colnade

        assert hasattr(colnade, "set_validation")
        assert hasattr(colnade, "is_validation_enabled")
        assert hasattr(colnade, "get_validation_level")


class TestValidationLevels:
    def teardown_method(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None

    def test_default_level_is_off(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {}, clear=True):
            assert get_validation_level() is ValidationLevel.OFF

    def test_set_level_off(self) -> None:
        set_validation("off")
        assert get_validation_level() is ValidationLevel.OFF
        assert is_validation_enabled() is False

    def test_set_level_structural(self) -> None:
        set_validation("structural")
        assert get_validation_level() is ValidationLevel.STRUCTURAL
        assert is_validation_enabled() is True

    def test_set_level_full(self) -> None:
        set_validation("full")
        assert get_validation_level() is ValidationLevel.FULL
        assert is_validation_enabled() is True

    def test_set_bool_true_maps_to_structural(self) -> None:
        set_validation(True)
        assert get_validation_level() is ValidationLevel.STRUCTURAL

    def test_set_bool_false_maps_to_off(self) -> None:
        set_validation(False)
        assert get_validation_level() is ValidationLevel.OFF

    def test_set_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid validation level"):
            set_validation("invalid")  # type: ignore[arg-type]

    def test_env_var_structural(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "structural"}):
            assert get_validation_level() is ValidationLevel.STRUCTURAL
            assert is_validation_enabled() is True

    def test_env_var_full(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "full"}):
            assert get_validation_level() is ValidationLevel.FULL
            assert is_validation_enabled() is True

    def test_env_var_off(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "off"}):
            assert get_validation_level() is ValidationLevel.OFF
            assert is_validation_enabled() is False

    def test_env_var_1_maps_to_structural(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "1"}):
            assert get_validation_level() is ValidationLevel.STRUCTURAL

    def test_env_var_FULL_case_insensitive(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "FULL"}):
            assert get_validation_level() is ValidationLevel.FULL

    def test_set_overrides_env_var_level(self) -> None:
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "full"}):
            set_validation("off")
            assert get_validation_level() is ValidationLevel.OFF


# ---------------------------------------------------------------------------
# Literal type checking
# ---------------------------------------------------------------------------


class ValUsers(Schema):
    id: Column[UInt64]
    name: Column[Utf8]


class TestCheckLiteralType:
    def setup_method(self) -> None:
        set_validation(True)

    def teardown_method(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None

    def test_int_for_uint64_ok(self) -> None:
        check_literal_type(42, UInt64)

    def test_float_for_uint64_raises(self) -> None:
        with pytest.raises(TypeError, match="float"):
            check_literal_type(1.0, UInt64)

    def test_str_for_utf8_ok(self) -> None:
        check_literal_type("hello", Utf8)

    def test_int_for_utf8_raises(self) -> None:
        with pytest.raises(TypeError, match="int"):
            check_literal_type(42, Utf8)

    def test_none_always_ok(self) -> None:
        check_literal_type(None, UInt64)

    def test_disabled_skips_check(self) -> None:
        set_validation(False)
        check_literal_type(1.0, UInt64)  # No error


class TestLiteralCheckInExpressions:
    def setup_method(self) -> None:
        set_validation(True)

    def teardown_method(self) -> None:
        import colnade.validation

        colnade.validation._validation_level = None

    def test_column_add_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="str.*UInt64"):
            ValUsers.id + "hello"

    def test_column_add_correct_type_ok(self) -> None:
        ValUsers.id + 1

    def test_column_gt_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="int.*Utf8"):
            _ = ValUsers.name > 42

    def test_column_gt_correct_type_ok(self) -> None:
        _ = ValUsers.name > "Alice"

    def test_fill_null_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="float"):
            ValUsers.id.fill_null(1.0)

    def test_fill_null_correct_type_ok(self) -> None:
        ValUsers.id.fill_null(0)

    def test_radd_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="str.*UInt64"):
            "hello" + ValUsers.id

    def test_disabled_no_error(self) -> None:
        set_validation(False)
        ValUsers.id + "hello"  # No error
