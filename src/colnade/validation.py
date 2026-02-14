"""Runtime schema validation toggle.

Validation is **off** by default for zero overhead in production.
Enable it via environment variable (ideal for CI) or programmatically::

    # Environment variable
    COLNADE_VALIDATE=1 pytest tests/

    # Programmatic
    import colnade
    colnade.set_validation(True)

When enabled, data boundaries (``read_parquet``, ``from_batches``, etc.)
automatically validate that loaded data matches the declared schema.

``DataFrame.validate()`` and ``LazyFrame.validate()`` always run
explicitly regardless of this toggle.
"""

from __future__ import annotations

import os

_validation_enabled: bool | None = None


def is_validation_enabled() -> bool:
    """Return whether automatic validation at data boundaries is enabled."""
    if _validation_enabled is not None:
        return _validation_enabled
    return os.environ.get("COLNADE_VALIDATE", "").lower() in ("1", "true", "yes")


def set_validation(enabled: bool) -> None:
    """Enable or disable automatic validation at data boundaries."""
    global _validation_enabled
    _validation_enabled = enabled
