"""Shared fixtures for end-to-end pipeline tests.

Provides parquet/CSV files with realistic data for testing full pipelines
through the Polars backend.
"""

from __future__ import annotations

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def _make_users(n: int = 100) -> pl.DataFrame:
    """Generate a users DataFrame with varied ages, scores, and some nulls in score."""
    import random

    random.seed(42)
    names = [f"user_{i:03d}" for i in range(1, n + 1)]
    ages = [random.randint(18, 65) for _ in range(n)]
    scores = [round(random.uniform(0, 100), 2) if i % 10 != 0 else None for i in range(n)]
    return pl.DataFrame(
        {
            "id": pl.Series(list(range(1, n + 1)), dtype=pl.UInt64),
            "name": names,
            "age": pl.Series(ages, dtype=pl.UInt64),
            "score": pl.Series(scores, dtype=pl.Float64),
        }
    )


def _make_orders(n: int = 200, max_user_id: int = 100) -> pl.DataFrame:
    """Generate an orders DataFrame linked to users via user_id."""
    import random

    random.seed(123)
    user_ids = [random.randint(1, max_user_id) for _ in range(n)]
    amounts = [round(random.uniform(10, 500), 2) for _ in range(n)]
    return pl.DataFrame(
        {
            "id": pl.Series(list(range(1, n + 1)), dtype=pl.UInt64),
            "user_id": pl.Series(user_ids, dtype=pl.UInt64),
            "amount": pl.Series(amounts, dtype=pl.Float64),
        }
    )


def _make_products(n: int = 50) -> pl.DataFrame:
    """Generate a products DataFrame."""
    import random

    random.seed(99)
    return pl.DataFrame(
        {
            "product_id": pl.Series(list(range(1, n + 1)), dtype=pl.UInt64),
            "product_name": [f"product_{i}" for i in range(1, n + 1)],
            "price": pl.Series(
                [round(random.uniform(5, 200), 2) for _ in range(n)],
                dtype=pl.Float64,
            ),
        }
    )


def _make_order_items(
    n: int = 300, max_order_id: int = 200, max_product_id: int = 50
) -> pl.DataFrame:
    """Generate order items linking orders to products."""
    import random

    random.seed(77)
    return pl.DataFrame(
        {
            "order_id": pl.Series(
                [random.randint(1, max_order_id) for _ in range(n)],
                dtype=pl.UInt64,
            ),
            "product_id": pl.Series(
                [random.randint(1, max_product_id) for _ in range(n)],
                dtype=pl.UInt64,
            ),
            "quantity": pl.Series([random.randint(1, 10) for _ in range(n)], dtype=pl.UInt64),
        }
    )


def _make_struct_users(n: int = 20) -> pl.DataFrame:
    """Generate users with struct address column."""
    streets = [f"{i * 100} Main St" for i in range(1, n + 1)]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"] * (n // 5 + 1)
    return pl.DataFrame(
        {
            "id": pl.Series(list(range(1, n + 1)), dtype=pl.UInt64),
            "name": [f"user_{i:03d}" for i in range(1, n + 1)],
            "address": pl.Series(
                [{"street": streets[i], "city": cities[i]} for i in range(n)],
                dtype=pl.Struct({"street": pl.String(), "city": pl.String()}),
            ),
        }
    )


def _make_list_users(n: int = 20) -> pl.DataFrame:
    """Generate users with list columns (tags and scores)."""
    import random

    random.seed(55)
    all_tags = ["admin", "user", "editor", "viewer", "manager"]
    tags = [random.sample(all_tags, k=random.randint(1, 3)) for _ in range(n)]
    score_lists = [
        [round(random.uniform(0, 100), 1) for _ in range(random.randint(1, 5))] for _ in range(n)
    ]
    return pl.DataFrame(
        {
            "id": pl.Series(list(range(1, n + 1)), dtype=pl.UInt64),
            "name": [f"user_{i:03d}" for i in range(1, n + 1)],
            "tags": pl.Series(tags, dtype=pl.List(pl.String())),
            "scores": pl.Series(score_lists, dtype=pl.List(pl.Float64())),
        }
    )


def _make_nullable_users(n: int = 50) -> pl.DataFrame:
    """Generate users with nulls in age and score for null handling tests."""
    import random

    random.seed(33)
    ages = [random.randint(18, 65) if i % 5 != 0 else None for i in range(n)]
    scores = [round(random.uniform(0, 100), 2) if i % 3 != 0 else None for i in range(n)]
    return pl.DataFrame(
        {
            "id": pl.Series(list(range(1, n + 1)), dtype=pl.UInt64),
            "name": [f"user_{i:03d}" for i in range(1, n + 1)],
            "age": pl.Series(ages, dtype=pl.UInt64),
            "score": pl.Series(scores, dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Parquet file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def users_parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "users.parquet")
    _make_users().write_parquet(path)
    return path


@pytest.fixture(scope="session")
def orders_parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "orders.parquet")
    _make_orders().write_parquet(path)
    return path


@pytest.fixture(scope="session")
def products_parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "products.parquet")
    _make_products().write_parquet(path)
    return path


@pytest.fixture(scope="session")
def order_items_parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "order_items.parquet")
    _make_order_items().write_parquet(path)
    return path


@pytest.fixture(scope="session")
def struct_users_parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "struct_users.parquet")
    _make_struct_users().write_parquet(path)
    return path


@pytest.fixture(scope="session")
def list_users_parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "list_users.parquet")
    _make_list_users().write_parquet(path)
    return path


@pytest.fixture(scope="session")
def nullable_users_parquet(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "nullable_users.parquet")
    _make_nullable_users().write_parquet(path)
    return path


@pytest.fixture(scope="session")
def users_csv(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = str(tmp_path_factory.mktemp("data") / "users.csv")
    _make_users().write_csv(path)
    return path
