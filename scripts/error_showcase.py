"""Intentional type errors for the error showcase screenshot."""

from colnade import Column, DataFrame, Float64, Schema, UInt64, Utf8, mapped_from
from colnade_polars import read_parquet

# --- Schema definitions ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]
    score: Column[Float64]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[Float64]


# --- Error 1: Misspelled column name ---

x = Users.naem


# --- Error 2: Schema mismatch at function boundary ---

df: DataFrame[Users] = read_parquet("users.parquet", Users)
wrong: DataFrame[Orders] = df


# --- Error 3: Nullability mismatch in mapped_from ---


class NullableSource(Schema):
    age: Column[UInt64 | None]


class Bad(Schema):
    age: Column[UInt64] = mapped_from(NullableSource.age)
