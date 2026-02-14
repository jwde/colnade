"""Unit tests for the runtime validation toggle."""

from __future__ import annotations

import os
from unittest.mock import patch

from colnade.validation import is_validation_enabled, set_validation


class TestValidationToggle:
    def teardown_method(self) -> None:
        # Reset global state after each test
        set_validation.__wrapped__ if hasattr(set_validation, "__wrapped__") else None
        import colnade.validation

        colnade.validation._validation_enabled = None

    def test_default_is_disabled(self) -> None:
        import colnade.validation

        colnade.validation._validation_enabled = None
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

        colnade.validation._validation_enabled = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "1"}):
            assert is_validation_enabled() is True

    def test_env_var_true(self) -> None:
        import colnade.validation

        colnade.validation._validation_enabled = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "true"}):
            assert is_validation_enabled() is True

    def test_env_var_yes(self) -> None:
        import colnade.validation

        colnade.validation._validation_enabled = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "yes"}):
            assert is_validation_enabled() is True

    def test_env_var_TRUE_case_insensitive(self) -> None:
        import colnade.validation

        colnade.validation._validation_enabled = None
        with patch.dict(os.environ, {"COLNADE_VALIDATE": "TRUE"}):
            assert is_validation_enabled() is True

    def test_env_var_invalid(self) -> None:
        import colnade.validation

        colnade.validation._validation_enabled = None
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
