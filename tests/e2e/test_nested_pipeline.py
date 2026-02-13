"""E2E: Nested types pipeline — struct field access and list operations.

Tests pipelines with struct columns (field access, filter on struct fields)
and list columns (len, get, contains, aggregation).
"""

from __future__ import annotations

from colnade import Column, DataFrame, Float64, List, Schema, Struct, UInt64, Utf8
from colnade_polars.io import read_parquet

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Address(Schema):
    street: Column[Utf8]
    city: Column[Utf8]


class UserProfile(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    address: Column[Struct[Address]]


class TaggedUsers(Schema):
    id: Column[UInt64]
    name: Column[Utf8]
    tags: Column[List[Utf8]]
    scores: Column[List[Float64]]


# ---------------------------------------------------------------------------
# Struct tests
# ---------------------------------------------------------------------------


class TestStructPipeline:
    def test_read_struct_data(self, struct_users_parquet: str) -> None:
        """Read parquet with struct columns preserves structure."""
        df = read_parquet(struct_users_parquet, UserProfile)
        assert isinstance(df, DataFrame)
        assert df._schema is UserProfile
        assert df._data.shape[0] == 20

    def test_filter_by_struct_field_comparison(self, struct_users_parquet: str) -> None:
        """Filter by struct field equality."""
        df = read_parquet(struct_users_parquet, UserProfile)
        result = df.filter(UserProfile.address.field(Address.city) == "New York")

        assert result._data.shape[0] > 0
        # All remaining rows should have city "New York"
        cities = result._data["address"].struct.field("city").to_list()
        assert all(c == "New York" for c in cities)

    def test_struct_field_not_null(self, struct_users_parquet: str) -> None:
        """Struct field access produces valid expression for filtering."""
        df = read_parquet(struct_users_parquet, UserProfile)
        # Struct field comparison with != produces valid filter
        result = df.filter(UserProfile.address.field(Address.city) != "New York")
        assert result._data.shape[0] > 0


# ---------------------------------------------------------------------------
# List tests
# ---------------------------------------------------------------------------


class TestListPipeline:
    def test_read_list_data(self, list_users_parquet: str) -> None:
        """Read parquet with list columns preserves structure."""
        df = read_parquet(list_users_parquet, TaggedUsers)
        assert isinstance(df, DataFrame)
        assert df._schema is TaggedUsers
        assert df._data.shape[0] == 20

    def test_filter_by_list_contains(self, list_users_parquet: str) -> None:
        """Filter users who have 'admin' tag."""
        df = read_parquet(list_users_parquet, TaggedUsers)
        result = df.filter(TaggedUsers.tags.list.contains("admin"))

        assert result._data.shape[0] > 0
        # All remaining rows should have "admin" in tags
        for tags in result._data["tags"].to_list():
            assert "admin" in tags

    def test_list_len(self, list_users_parquet: str) -> None:
        """Filter by list length."""
        df = read_parquet(list_users_parquet, TaggedUsers)
        result = df.filter(TaggedUsers.tags.list.len() > 1)

        assert result._data.shape[0] > 0
        for tags in result._data["tags"].to_list():
            assert len(tags) > 1

    def test_list_sum_in_with_columns(self, list_users_parquet: str) -> None:
        """Use list.sum() in with_columns to compute total score."""
        df = read_parquet(list_users_parquet, TaggedUsers)
        result = df.with_columns(TaggedUsers.scores.list.sum().alias(TaggedUsers.scores))

        # scores column should now contain scalar sums
        assert result._data.shape[0] == 20
