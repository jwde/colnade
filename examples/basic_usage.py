"""Basic usage: schema definition, filter, select, aggregate.

Demonstrates the core Colnade workflow with the Polars backend.
"""

from __future__ import annotations

import tempfile

from colnade import Column, Float64, Schema, UInt64, Utf8
from colnade_polars import from_dict, write_parquet
from colnade_polars.io import read_parquet

# ---------------------------------------------------------------------------
# 1. Define schemas — typed column references, verified by the type checker
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class UserSummary(Schema):
    name: Column[Utf8]
    score: Column[Float64]


# ---------------------------------------------------------------------------
# 2. Create sample data and write to Parquet
# ---------------------------------------------------------------------------

df = from_dict(
    Users,
    {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [30, 25, 35, 28, 40],
        "score": [85.0, 92.5, 78.0, 95.0, 88.0],
    },
)

with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
    write_parquet(df, f.name)
    parquet_path = f.name

# ---------------------------------------------------------------------------
# 3. Read typed data — the type checker knows df is DataFrame[Users]
# ---------------------------------------------------------------------------

df = read_parquet(parquet_path, Users)
print(f"Read {df._data.shape[0]} users")
schema_name = df._schema.__name__ if df._schema else "None"
print(f"Schema: {schema_name}")
print()

# ---------------------------------------------------------------------------
# 4. Filter — column references are attributes, not strings
# ---------------------------------------------------------------------------

# Users.age is a Column[UInt64] descriptor — misspelling it is a type error
adults = df.filter(Users.age >= 30)
print("Users aged 30+:")
print(adults._data)
print()

# ---------------------------------------------------------------------------
# 5. Sort with typed sort expressions
# ---------------------------------------------------------------------------

by_score = df.sort(Users.score.desc())
print("Users by score (descending):")
print(by_score._data)
print()

# ---------------------------------------------------------------------------
# 6. with_columns — compute new values from typed expressions
# ---------------------------------------------------------------------------

doubled = df.with_columns((Users.score * 2).alias(Users.score))
print("Doubled scores:")
print(doubled._data)
print()

# ---------------------------------------------------------------------------
# 7. Select + cast_schema — bind to an output schema
# ---------------------------------------------------------------------------

summary = df.select(Users.name, Users.score).cast_schema(UserSummary)
summary_name = summary._schema.__name__ if summary._schema else "None"
print(f"Summary schema: {summary_name}")
print(f"Summary columns: {summary._data.columns}")
print(summary._data)
print()

# ---------------------------------------------------------------------------
# 8. Write result
# ---------------------------------------------------------------------------

with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
    write_parquet(summary, f.name)
    print(f"Wrote summary to {f.name}")

print("\nDone!")
