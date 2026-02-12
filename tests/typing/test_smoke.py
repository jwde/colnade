"""Smoke test for type checker CI integration.

This file must pass ty check with zero errors.
"""

import colnade
import colnade_polars


def check_imports() -> None:
    _ = colnade
    _ = colnade_polars
