"""Backend protocols (what adapters must implement)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from colnade.expr import AliasedExpr, Expr, JoinCondition, SortExpr
    from colnade.schema import Column, Schema


class BackendProtocol(Protocol):
    """Interface that all backend adapters must implement.

    Each method takes backend-native data (``Any``) as ``source`` and returns
    backend-native data. The DataFrame/LazyFrame layer wraps results in typed
    frame instances.
    """

    # --- Expression translation ---

    def translate_expr(self, expr: Expr[Any]) -> Any:
        """Translate a Colnade expression AST to a backend-native expression."""
        ...

    # --- Schema-preserving operations ---

    def filter(self, source: Any, predicate: Expr[Any]) -> Any: ...

    def sort(
        self,
        source: Any,
        by: Sequence[Column[Any] | SortExpr],
        descending: bool,
    ) -> Any: ...

    def limit(self, source: Any, n: int) -> Any: ...

    def head(self, source: Any, n: int) -> Any: ...

    def tail(self, source: Any, n: int) -> Any: ...

    def sample(self, source: Any, n: int) -> Any: ...

    def unique(self, source: Any, columns: Sequence[Column[Any]]) -> Any: ...

    def drop_nulls(self, source: Any, columns: Sequence[Column[Any]]) -> Any: ...

    def with_columns(self, source: Any, exprs: Sequence[AliasedExpr[Any] | Expr[Any]]) -> Any: ...

    def concat(self, sources: Sequence[Any]) -> Any: ...

    # --- Schema-transforming operations ---

    def select(self, source: Any, columns: Sequence[Column[Any]]) -> Any: ...

    def group_by_agg(
        self,
        source: Any,
        keys: Sequence[Column[Any]],
        aggs: Sequence[AliasedExpr[Any]],
    ) -> Any: ...

    def agg(self, source: Any, aggs: Sequence[AliasedExpr[Any]]) -> Any: ...

    def join(self, left: Any, right: Any, on: JoinCondition, how: str) -> Any: ...

    def cast_schema(self, source: Any, column_mapping: dict[str, str]) -> Any: ...

    # --- Lazy / collect ---

    def lazy(self, source: Any) -> Any: ...

    def collect(self, source: Any) -> Any: ...

    # --- Validation ---

    def validate_schema(self, source: Any, schema: type[Schema]) -> None: ...

    def validate_field_constraints(self, source: Any, schema: type[Schema]) -> None: ...

    # --- Arrow boundary ---

    def to_arrow_batches(
        self,
        source: Any,
        batch_size: int | None,
    ) -> Iterator[Any]: ...

    def from_arrow_batches(
        self,
        batches: Iterator[Any],
        schema: type[Schema],
    ) -> Any: ...

    # --- Construction ---

    def from_dict(
        self,
        data: dict[str, Sequence[Any]],
        schema: type[Schema],
    ) -> Any:
        """Create a backend-native data object from a columnar dict.

        The backend reads column dtypes from ``schema`` and coerces values
        to the correct native types.
        """
        ...

    # --- Row access ---

    def row_count(self, source: Any) -> int: ...

    def iter_row_dicts(self, source: Any) -> Iterator[dict[str, Any]]: ...
