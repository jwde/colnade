"""Unit tests for cast_schema, mapped_from, and SchemaError."""

from __future__ import annotations

import pytest

from colnade import (
    Column,
    DataFrame,
    JoinedDataFrame,
    JoinedLazyFrame,
    LazyFrame,
    Schema,
    SchemaError,
    UInt64,
    Utf8,
    mapped_from,
)
from colnade.schema import _MappedFrom

# ---------------------------------------------------------------------------
# Minimal mock backend — returns source data unchanged for all operations
# ---------------------------------------------------------------------------


class _MockBackend:
    """Backend stub for unit tests. Returns the first positional arg (source)."""

    def __getattr__(self, name: str):  # noqa: ANN204
        def _method(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return args[0] if args else None

        return _method


_BACKEND = _MockBackend()


# ---------------------------------------------------------------------------
# Test fixture schemas
# ---------------------------------------------------------------------------


class Users(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    age: Column[UInt64]


class Orders(Schema):
    id: Column[UInt64]
    user_id: Column[UInt64]
    amount: Column[UInt64]


# Target schemas for cast_schema tests


class UsersSummary(Schema):
    """Same column names as Users — name-matching should resolve."""

    id: Column[UInt64]
    name: Column[Utf8]


class RenamedUsers(Schema):
    """Uses mapped_from to rename columns."""

    user_id: Column[UInt64] = mapped_from(Users.id)
    user_name: Column[Utf8] = mapped_from(Users.name)


class MixedMapping(Schema):
    """Mix of mapped_from and name-matched columns."""

    id: Column[UInt64]
    user_name: Column[Utf8] = mapped_from(Users.name)


class JoinTarget(Schema):
    """Target for joined cast_schema — columns from both schemas."""

    user_name: Column[Utf8] = mapped_from(Users.name)
    user_id: Column[UInt64] = mapped_from(Orders.user_id)
    amount: Column[UInt64]


class MissingColumn(Schema):
    """Has a column that doesn't exist in Users."""

    id: Column[UInt64]
    email: Column[Utf8]


class ExtraCheck(Schema):
    """Only one column — 'name' and 'age' would be extra."""

    id: Column[UInt64]


# ---------------------------------------------------------------------------
# _MappedFrom and mapped_from()
# ---------------------------------------------------------------------------


class TestMappedFrom:
    def test_mapped_from_returns_mapped_from_instance(self) -> None:
        result = mapped_from(Users.name)
        assert isinstance(result, _MappedFrom)

    def test_mapped_from_stores_source(self) -> None:
        result = mapped_from(Users.name)
        assert isinstance(result, _MappedFrom)
        assert result.source is Users.name

    def test_schema_with_mapped_from_has_mapped_from_on_column(self) -> None:
        assert RenamedUsers.user_id._mapped_from is Users.id
        assert RenamedUsers.user_name._mapped_from is Users.name

    def test_schema_without_mapped_from_has_none(self) -> None:
        assert Users.id._mapped_from is None
        assert Users.name._mapped_from is None
        assert Users.age._mapped_from is None

    def test_mixed_schema_mapped_from(self) -> None:
        assert MixedMapping.id._mapped_from is None
        assert MixedMapping.user_name._mapped_from is Users.name


# ---------------------------------------------------------------------------
# SchemaError
# ---------------------------------------------------------------------------


class TestSchemaError:
    def test_missing_columns_attribute(self) -> None:
        err = SchemaError(missing_columns=["email", "phone"])
        assert err.missing_columns == ["email", "phone"]

    def test_extra_columns_attribute(self) -> None:
        err = SchemaError(extra_columns=["age", "name"])
        assert err.extra_columns == ["age", "name"]

    def test_type_mismatches_attribute(self) -> None:
        err = SchemaError(type_mismatches={"age": ("UInt64", "Utf8")})
        assert err.type_mismatches == {"age": ("UInt64", "Utf8")}

    def test_null_violations_attribute(self) -> None:
        err = SchemaError(null_violations=["name"])
        assert err.null_violations == ["name"]

    def test_message_contains_missing(self) -> None:
        err = SchemaError(missing_columns=["email"])
        assert "Missing columns" in str(err)
        assert "email" in str(err)

    def test_message_contains_extra(self) -> None:
        err = SchemaError(extra_columns=["age"])
        assert "Extra columns" in str(err)
        assert "age" in str(err)

    def test_default_message(self) -> None:
        err = SchemaError()
        assert "Schema validation failed" in str(err)

    def test_is_exception(self) -> None:
        assert issubclass(SchemaError, Exception)


# ---------------------------------------------------------------------------
# DataFrame.cast_schema — name matching
# ---------------------------------------------------------------------------


class TestCastSchemaNameMatching:
    def setup_method(self) -> None:
        self.df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)

    def test_name_match_succeeds(self) -> None:
        result = self.df.cast_schema(UsersSummary)
        assert isinstance(result, DataFrame)
        assert result._schema is UsersSummary

    def test_missing_column_raises(self) -> None:
        with pytest.raises(SchemaError) as exc_info:
            self.df.cast_schema(MissingColumn)
        assert "email" in exc_info.value.missing_columns

    def test_extra_columns_drop_succeeds(self) -> None:
        result = self.df.cast_schema(ExtraCheck, extra="drop")
        assert isinstance(result, DataFrame)
        assert result._schema is ExtraCheck

    def test_extra_columns_forbid_raises(self) -> None:
        with pytest.raises(SchemaError) as exc_info:
            self.df.cast_schema(ExtraCheck, extra="forbid")
        assert "name" in exc_info.value.extra_columns
        assert "age" in exc_info.value.extra_columns


# ---------------------------------------------------------------------------
# DataFrame.cast_schema — mapped_from
# ---------------------------------------------------------------------------


class TestCastSchemaMappedFrom:
    def setup_method(self) -> None:
        self.df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)

    def test_mapped_from_resolves(self) -> None:
        result = self.df.cast_schema(RenamedUsers)
        assert isinstance(result, DataFrame)
        assert result._schema is RenamedUsers

    def test_mixed_mapped_from_and_name_match(self) -> None:
        result = self.df.cast_schema(MixedMapping)
        assert isinstance(result, DataFrame)
        assert result._schema is MixedMapping


# ---------------------------------------------------------------------------
# DataFrame.cast_schema — explicit mapping
# ---------------------------------------------------------------------------


class TestCastSchemaExplicitMapping:
    def setup_method(self) -> None:
        self.df: DataFrame[Users] = DataFrame(_schema=Users, _backend=_BACKEND)

    def test_explicit_mapping_resolves(self) -> None:
        result = self.df.cast_schema(
            MissingColumn,
            mapping={MissingColumn.email: Users.name},
        )
        assert isinstance(result, DataFrame)
        assert result._schema is MissingColumn

    def test_explicit_mapping_overrides_mapped_from(self) -> None:
        # Override RenamedUsers.user_id (mapped_from Users.id) to point at Users.age
        result = self.df.cast_schema(
            RenamedUsers,
            mapping={RenamedUsers.user_id: Users.age},
        )
        assert isinstance(result, DataFrame)
        assert result._schema is RenamedUsers


# ---------------------------------------------------------------------------
# cast_schema on JoinedDataFrame
# ---------------------------------------------------------------------------


class TestCastSchemaOnJoined:
    def setup_method(self) -> None:
        self.joined = JoinedDataFrame(_schema_left=Users, _schema_right=Orders, _backend=_BACKEND)

    def test_name_match_across_both_schemas(self) -> None:
        result = self.joined.cast_schema(JoinTarget)
        assert isinstance(result, DataFrame)
        assert result._schema is JoinTarget

    def test_ambiguous_name_without_mapping_raises(self) -> None:
        """'id' exists in both Users and Orders — should be ambiguous."""

        class AmbiguousTarget(Schema):
            id: Column[UInt64]

        with pytest.raises(SchemaError) as exc_info:
            self.joined.cast_schema(AmbiguousTarget)
        assert "id" in exc_info.value.missing_columns

    def test_ambiguous_name_with_explicit_mapping_succeeds(self) -> None:
        class AmbiguousTarget(Schema):
            id: Column[UInt64]

        result = self.joined.cast_schema(
            AmbiguousTarget,
            mapping={AmbiguousTarget.id: Users.id},
        )
        assert isinstance(result, DataFrame)
        assert result._schema is AmbiguousTarget

    def test_ambiguous_name_with_mapped_from_succeeds(self) -> None:
        class AmbiguousResolved(Schema):
            id: Column[UInt64] = mapped_from(Users.id)

        result = self.joined.cast_schema(AmbiguousResolved)
        assert isinstance(result, DataFrame)
        assert result._schema is AmbiguousResolved

    def test_returns_dataframe_not_joined(self) -> None:
        result = self.joined.cast_schema(JoinTarget)
        assert isinstance(result, DataFrame)
        assert not isinstance(result, JoinedDataFrame)


# ---------------------------------------------------------------------------
# cast_schema on LazyFrame
# ---------------------------------------------------------------------------


class TestCastSchemaOnLazy:
    def test_lazyframe_cast_schema_returns_lazyframe(self) -> None:
        lf: LazyFrame[Users] = LazyFrame(_schema=Users, _backend=_BACKEND)
        result = lf.cast_schema(UsersSummary)
        assert isinstance(result, LazyFrame)
        assert result._schema is UsersSummary

    def test_joined_lazyframe_cast_schema_returns_lazyframe(self) -> None:
        jlf = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders, _backend=_BACKEND)
        result = jlf.cast_schema(JoinTarget)
        assert isinstance(result, LazyFrame)
        assert result._schema is JoinTarget

    def test_joined_lazyframe_not_joined_after_cast(self) -> None:
        jlf = JoinedLazyFrame(_schema_left=Users, _schema_right=Orders, _backend=_BACKEND)
        result = jlf.cast_schema(JoinTarget)
        assert not isinstance(result, JoinedLazyFrame)


# ---------------------------------------------------------------------------
# cast_schema on untyped (None schema) DataFrame
# ---------------------------------------------------------------------------


class TestCastSchemaOnUntyped:
    def test_none_schema_skips_validation(self) -> None:
        """DataFrame[Any] from select() has _schema=None — cast_schema just sets schema."""
        df: DataFrame[Users] = DataFrame(_schema=None, _backend=_BACKEND)
        result = df.cast_schema(UsersSummary)
        assert isinstance(result, DataFrame)
        assert result._schema is UsersSummary
