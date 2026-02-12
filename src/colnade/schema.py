"""Schema base class, Column descriptors, and schema metaclass.

Defines the foundational layer that makes column references statically verifiable:
- ``SchemaMeta`` — metaclass that creates Column descriptors from annotations
- ``Schema`` — base class for user-defined schemas (extends Protocol)
- ``Column[DType, SchemaType]`` — typed column descriptor
"""

from __future__ import annotations

import typing
from typing import Any, Generic, Protocol, TypeVar

from colnade._types import DType

# ---------------------------------------------------------------------------
# Schema-bound TypeVars (must live here because they reference Schema)
# ---------------------------------------------------------------------------

# Primary schema TypeVar — used in DataFrame[S], Column[DType, S], etc.
S = TypeVar("S", bound="Schema")

# Additional schema TypeVars for joins and multi-schema operations
S2 = TypeVar("S2", bound="Schema")
S3 = TypeVar("S3", bound="Schema")

# SchemaType alias — for Column's second type parameter
SchemaType = TypeVar("SchemaType", bound="Schema")

# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

_schema_registry: dict[str, type[Schema]] = {}

# ---------------------------------------------------------------------------
# Column descriptor
# ---------------------------------------------------------------------------


class Column(Generic[DType, SchemaType]):
    """A typed reference to a named column in a schema.

    At the type level: ``Column[UInt8, Users]`` tells the type checker that
    this column holds ``UInt8`` data and belongs to the ``Users`` schema.

    At runtime: stores the column ``name``, ``dtype`` annotation, and owning
    ``schema`` class. Expression-building methods are wired in issue #4.
    """

    __slots__ = ("name", "dtype", "schema")

    def __init__(self, name: str, dtype: Any, schema: type) -> None:
        self.name = name
        self.dtype = dtype
        self.schema = schema

    def __repr__(self) -> str:
        return f"Column({self.name!r}, dtype={self.dtype}, schema={self.schema.__name__})"


# ---------------------------------------------------------------------------
# Schema metaclass
# ---------------------------------------------------------------------------


class SchemaMeta(type(Protocol)):  # type: ignore[misc]
    """Metaclass for Schema that creates Column descriptors from annotations.

    At class creation time:
    1. Collects annotations from the class and all bases (MRO traversal).
    2. Creates Column descriptor objects for each non-private field.
    3. Stores column descriptors in ``cls._columns``.
    4. Registers the schema in the internal registry.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> SchemaMeta:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Resolve annotations via get_type_hints() to handle PEP 563
        # (from __future__ import annotations) which stores annotations as strings.
        # get_type_hints() traverses the MRO and resolves forward references.
        try:
            annotations: dict[str, Any] = typing.get_type_hints(cls, include_extras=True)
        except Exception:
            # Fallback: collect raw annotations from MRO (base-first so children override)
            annotations = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, "__annotations__", {}))

        # Build Column descriptors for non-private annotations
        columns: dict[str, Column[Any, Any]] = {}
        for col_name, col_type in annotations.items():
            if col_name.startswith("_"):
                continue
            descriptor: Column[Any, Any] = Column(name=col_name, dtype=col_type, schema=cls)
            setattr(cls, col_name, descriptor)
            columns[col_name] = descriptor

        cls._columns = columns  # type: ignore[attr-defined]

        # Register non-base schemas
        if name != "Schema":
            _schema_registry[name] = cls  # type: ignore[assignment]

        return cls  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Schema base class
# ---------------------------------------------------------------------------


class Schema(Protocol, metaclass=SchemaMeta):
    """Base class for user-defined data schemas.

    Subclass this to define a typed schema::

        class Users(Schema):
            id: UInt64
            name: Utf8
            age: UInt8 | None

    The metaclass replaces each annotation with a ``Column`` descriptor,
    enabling typed column references: ``Users.age`` → ``Column[UInt8 | None, Users]``.
    """

    # Populated by SchemaMeta; declared here for type checker visibility
    _columns: dict[str, Column[Any, Any]]
