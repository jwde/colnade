"""Verify CHANGELOG.md dates are in descending chronological order.

Prevents a recurring issue where release entries get the wrong date,
making newer versions appear older than previous ones.

    uv run python scripts/check_changelog.py
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
HEADING_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$")


def main() -> int:
    entries: list[tuple[str, date]] = []
    for line in CHANGELOG.read_text().splitlines():
        m = HEADING_RE.match(line)
        if m:
            entries.append((m.group(1), date.fromisoformat(m.group(2))))

    if not entries:
        print("ERROR: no versioned entries found in CHANGELOG.md")
        return 1

    errors = 0
    for i in range(len(entries) - 1):
        ver_a, date_a = entries[i]
        ver_b, date_b = entries[i + 1]
        if date_a < date_b:
            print(
                f"ERROR: v{ver_a} ({date_a}) is dated earlier than "
                f"v{ver_b} ({date_b}) — dates must be in descending order"
            )
            errors += 1

    if errors:
        return 1

    print(f"OK: {len(entries)} changelog entries in chronological order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
