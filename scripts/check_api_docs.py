#!/usr/bin/env python3
"""Verify every public export in __all__ has a ::: directive in the API docs.

Usage:
    uv run python scripts/check_api_docs.py

Exits 0 if all exports are documented, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS_API_DIR = Path(__file__).resolve().parent.parent / "docs" / "api"
DIRECTIVE_RE = re.compile(r"^:::\ +(\S+)", re.MULTILINE)

PACKAGES: dict[str, str] = {
    "colnade": "src/colnade/__init__.py",
    "colnade_polars": "colnade-polars/src/colnade_polars/__init__.py",
    "colnade_pandas": "colnade-pandas/src/colnade_pandas/__init__.py",
    "colnade_dask": "colnade-dask/src/colnade_dask/__init__.py",
}


def _collect_documented_symbols() -> set[str]:
    """Scan docs/api/*.md for ::: directives and return the set of documented paths."""
    documented: set[str] = set()
    for md_file in DOCS_API_DIR.glob("*.md"):
        text = md_file.read_text()
        for match in DIRECTIVE_RE.finditer(text):
            documented.add(match.group(1))
    return documented


def _resolve_symbol_path(package_name: str, symbol_name: str, symbol: object) -> str:
    """Resolve a symbol to its full module.qualname path for matching ::: directives."""
    module = getattr(symbol, "__module__", None)
    qualname = getattr(symbol, "__qualname__", symbol_name)
    if module:
        return f"{module}.{qualname}"
    return f"{package_name}.{symbol_name}"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "colnade-polars" / "src"))
    sys.path.insert(0, str(root / "colnade-pandas" / "src"))
    sys.path.insert(0, str(root / "colnade-dask" / "src"))

    missing: list[tuple[str, str]] = []

    for package_name, _init_path in PACKAGES.items():
        mod = __import__(package_name)
        all_exports: list[str] = getattr(mod, "__all__", [])

        documented = _collect_documented_symbols()

        for name in all_exports:
            symbol = getattr(mod, name, None)
            if symbol is None:
                missing.append((package_name, f"{package_name}.{name} (not importable)"))
                continue

            path = _resolve_symbol_path(package_name, name, symbol)
            if path not in documented:
                missing.append((package_name, path))

    if missing:
        print("API docs coverage check FAILED.\n")
        print("The following public exports are missing ::: directives in docs/api/*.md:\n")
        for pkg, path in missing:
            print(f"  - {path}  (from {pkg}.__all__)")
        print("\nAdd a '::: <path>' directive to the appropriate docs/api/*.md file.")
        return 1

    print("API docs coverage check passed. All exports documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
