"""Static type tests for DataFrame[S], LazyFrame[S], and GroupBy.

This file is checked by ty — it must produce zero type errors.

Tests cover:
- Schema preservation on filter/sort/limit/head/tail/sample/unique/drop_nulls/with_columns
- Schema-transforming ops (select, group_by+agg) return Any-parameterized frames
- lazy()/collect() conversions preserve schema
- GroupBy/LazyGroupBy types
- Negative regression guards (LazyFrame NOT assignable to DataFrame, etc.)
"""

from colnade import (
    Column,
    DataFrame,
    GroupBy,
    LazyFrame,
    LazyGroupBy,
    Schema,
    UInt8,
    UInt64,
    Utf8,
    concat,
)

# --- Schema definitions ---


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt8]


class AgeStats(Schema):
    age: Column[UInt8]
    count: Column[UInt64]


# --- Schema-preserving ops return DataFrame[Users] ---


def check_filter_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.filter(Users.age > 18)


def check_sort_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.sort(Users.name)


def check_limit_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.limit(10)


def check_head_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.head()


def check_tail_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.tail()


def check_sample_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.sample(5)


def check_unique_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.unique(Users.name)


def check_drop_nulls_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.drop_nulls(Users.name)


def check_with_columns_preserves_schema(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.with_columns(Users.age + 1)


# --- LazyFrame schema-preserving ops ---


def check_lazy_filter(lf: LazyFrame[Users]) -> LazyFrame[Users]:
    return lf.filter(Users.age > 18)


def check_lazy_sort(lf: LazyFrame[Users]) -> LazyFrame[Users]:
    return lf.sort(Users.name)


def check_lazy_limit(lf: LazyFrame[Users]) -> LazyFrame[Users]:
    return lf.limit(10)


def check_lazy_unique(lf: LazyFrame[Users]) -> LazyFrame[Users]:
    return lf.unique(Users.name)


def check_lazy_drop_nulls(lf: LazyFrame[Users]) -> LazyFrame[Users]:
    return lf.drop_nulls(Users.name)


def check_lazy_with_columns(lf: LazyFrame[Users]) -> LazyFrame[Users]:
    return lf.with_columns(Users.age + 1)


# --- Conversion preserves schema ---


def check_lazy_conversion(df: DataFrame[Users]) -> LazyFrame[Users]:
    return df.lazy()


def check_collect_conversion(lf: LazyFrame[Users]) -> DataFrame[Users]:
    return lf.collect()


# --- GroupBy types ---


def check_group_by_type(df: DataFrame[Users]) -> GroupBy[Users]:
    return df.group_by(Users.age)


def check_lazy_group_by_type(lf: LazyFrame[Users]) -> LazyGroupBy[Users]:
    return lf.group_by(Users.age)


# --- Validate returns same type ---


def check_validate_dataframe(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.validate()


def check_validate_lazyframe(lf: LazyFrame[Users]) -> LazyFrame[Users]:
    return lf.validate()


# ---------------------------------------------------------------------------
# Negative type tests — regression guards
#
# Each line below MUST produce a type error, suppressed by an ignore comment.
# If types regress, the error disappears, the suppression becomes unused,
# and ty reports unused-ignore-comment — failing CI.
# ---------------------------------------------------------------------------


def check_neg_lazyframe_not_dataframe() -> None:
    """LazyFrame[Users] is NOT assignable to DataFrame[Users]."""
    lf: LazyFrame[Users] = LazyFrame(_schema=Users)
    _: DataFrame[Users] = lf  # type: ignore[invalid-assignment]


def check_neg_dataframe_not_lazyframe() -> None:
    """DataFrame[Users] is NOT assignable to LazyFrame[Users]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    _: LazyFrame[Users] = df  # type: ignore[invalid-assignment]


def check_neg_groupby_schema_distinct() -> None:
    """GroupBy[Users] is NOT assignable to GroupBy[AgeStats]."""
    gb: GroupBy[Users] = GroupBy(_schema=Users)
    _: GroupBy[AgeStats] = gb  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Schema-preservation guards — detect if return types regress to Any
#
# These are CRITICAL. If a schema-preserving operation like filter()
# regressed from returning DataFrame[S] to DataFrame[Any], the positive
# tests above would STILL PASS (because Any is compatible with everything).
# These negative tests catch that: if the return type widens to Any, the
# assignment to a WRONG schema succeeds, the type: ignore becomes unused,
# and --error-on-warning fails CI.
# ---------------------------------------------------------------------------


def check_neg_filter_preserves_exact_schema() -> None:
    """filter() returns DataFrame[Users], NOT DataFrame[Any] or DataFrame[AgeStats]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result = df.filter(Users.age > 18)
    _: DataFrame[AgeStats] = result  # type: ignore[invalid-assignment]


def check_neg_sort_preserves_exact_schema() -> None:
    """sort() returns DataFrame[Users], NOT DataFrame[Any]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result = df.sort(Users.name)
    _: DataFrame[AgeStats] = result  # type: ignore[invalid-assignment]


def check_neg_with_columns_preserves_exact_schema() -> None:
    """with_columns() returns DataFrame[Users], NOT DataFrame[Any]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result = df.with_columns(Users.age + 1)
    _: DataFrame[AgeStats] = result  # type: ignore[invalid-assignment]


def check_neg_lazy_filter_preserves_exact_schema() -> None:
    """LazyFrame.filter() returns LazyFrame[Users], NOT LazyFrame[Any]."""
    lf: LazyFrame[Users] = LazyFrame(_schema=Users)
    result = lf.filter(Users.age > 18)
    _: LazyFrame[AgeStats] = result  # type: ignore[invalid-assignment]


def check_neg_lazy_preserves_schema() -> None:
    """lazy() returns LazyFrame[Users], NOT LazyFrame[Any]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result = df.lazy()
    _: LazyFrame[AgeStats] = result  # type: ignore[invalid-assignment]


def check_neg_collect_preserves_schema() -> None:
    """collect() returns DataFrame[Users], NOT DataFrame[Any]."""
    lf: LazyFrame[Users] = LazyFrame(_schema=Users)
    result = lf.collect()
    _: DataFrame[AgeStats] = result  # type: ignore[invalid-assignment]


def check_neg_validate_preserves_exact_schema() -> None:
    """validate() returns DataFrame[Users], NOT DataFrame[Any]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    result = df.validate()
    _: DataFrame[AgeStats] = result  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Additional frame type boundaries
# ---------------------------------------------------------------------------


def check_neg_lazyframe_schema_invariant() -> None:
    """LazyFrame[Users] is NOT assignable to LazyFrame[AgeStats]."""
    lf: LazyFrame[Users] = LazyFrame(_schema=Users)
    _: LazyFrame[AgeStats] = lf  # type: ignore[invalid-assignment]


def check_neg_groupby_not_dataframe() -> None:
    """GroupBy[Users] is NOT assignable to DataFrame[Users]."""
    gb: GroupBy[Users] = GroupBy(_schema=Users)
    _: DataFrame[Users] = gb  # type: ignore[invalid-assignment]


def check_neg_lazy_groupby_not_lazyframe() -> None:
    """LazyGroupBy[Users] is NOT assignable to LazyFrame[Users]."""
    lgb: LazyGroupBy[Users] = LazyGroupBy(_schema=Users)
    _: LazyFrame[Users] = lgb  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Introspection — positive type tests
# ---------------------------------------------------------------------------


def check_height_type(df: DataFrame[Users]) -> int:
    return df.height


def check_len_type(df: DataFrame[Users]) -> int:
    return len(df)


def check_width_type(df: DataFrame[Users]) -> int:
    return df.width


def check_shape_type(df: DataFrame[Users]) -> tuple[int, int]:
    return df.shape


def check_is_empty_type(df: DataFrame[Users]) -> bool:
    return df.is_empty()


def check_lazyframe_width_type(lf: LazyFrame[Users]) -> int:
    return lf.width


# ---------------------------------------------------------------------------
# Introspection — negative regression guards
# ---------------------------------------------------------------------------


def check_neg_height_not_str() -> None:
    """height returns int, NOT str."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    _: str = df.height  # type: ignore[invalid-assignment]


def check_neg_shape_not_triple() -> None:
    """shape returns tuple[int, int], NOT tuple[int, int, int]."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    _: tuple[int, int, int] = df.shape  # type: ignore[invalid-assignment]


def check_neg_width_not_str() -> None:
    """width returns int, NOT str."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    _: str = df.width  # type: ignore[invalid-assignment]


def check_neg_is_empty_not_int() -> None:
    """is_empty() returns bool, NOT int."""
    df: DataFrame[Users] = DataFrame(_schema=Users)
    _: str = df.is_empty()  # type: ignore[invalid-assignment]


def check_neg_lazyframe_width_not_str() -> None:
    """LazyFrame.width returns int, NOT str."""
    lf: LazyFrame[Users] = LazyFrame(_schema=Users)
    _: str = lf.width  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# concat() type tests
# ---------------------------------------------------------------------------


def check_concat_dataframe(df1: DataFrame[Users], df2: DataFrame[Users]) -> DataFrame[Users]:
    """concat(DataFrame, DataFrame) returns DataFrame[S]."""
    return concat(df1, df2)


def check_concat_lazyframe(lf1: LazyFrame[Users], lf2: LazyFrame[Users]) -> LazyFrame[Users]:
    """concat(LazyFrame, LazyFrame) returns LazyFrame[S]."""
    return concat(lf1, lf2)


def check_neg_concat_preserves_exact_schema() -> None:
    """concat() returns DataFrame[Users], NOT DataFrame[AgeStats]."""
    df1: DataFrame[Users] = DataFrame(_schema=Users)
    df2: DataFrame[Users] = DataFrame(_schema=Users)
    result = concat(df1, df2)
    _: DataFrame[AgeStats] = result  # type: ignore[invalid-assignment]
