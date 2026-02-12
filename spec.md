# Colnade: A Statically Type-Safe DataFrame Abstraction Layer

## Project Specification v0.1

---

## 1. Executive Summary

### 1.1 Problem

Python's DataFrame libraries (Polars, Pandas, Spark, etc.) offer no meaningful static type safety. Column references are runtime strings (`pl.col("name")`), schema information is invisible to type checkers, and column misspellings, type mismatches, and schema violations are only caught at runtime. This is in stark contrast to equivalent `list[dataclass]` representations, where attribute access is fully verifiable by tools like `ty`, `mypy`, and `pyright`.

Existing attempts to solve this fall into two categories, both insufficient:

- **Runtime validation** (Pandera, Patito, dataframely): Define schemas and validate at runtime. Useful, but the type checker has no visibility into column references or schema transformations within function bodies.
- **Static generics** (StaticFrame): Encode column types positionally in `TypeVarTuple` generics. Columns are unnamed at the type level, schema transformations aren't tracked, and it requires adopting a niche DataFrame engine.

A `TypeVarDict` proposal on the `python/typing` repo would solve this at the language level, but it has stalled with no PEP or implementation timeline.

### 1.2 Solution

Colnade is a **statically type-safe abstraction layer** over existing DataFrame engines. It provides:

- **Schema classes** that extend `Protocol`, enabling structural subtyping natively in all major type checkers.
- **Typed column descriptors** that replace string-based column references (`Users.age` instead of `pl.col("age")`), making column access statically verifiable via standard attribute resolution.
- **A typed expression DSL** where operations on column descriptors produce typed expression trees (`Users.age > 18` returns `Expr[Bool]`), with type-incorrect operations (e.g., `.sum()` on a string column) caught at lint time.
- **Schema-preserving generics** using bounded `TypeVar` and `Protocol` structural subtyping, enabling generic utility functions that preserve the full schema of their input.
- **Backend adapters** that lower the abstract expression tree into engine-native calls (Polars, Snowpark, Ray, DuckDB, etc.), so the same typed pipeline code runs on any engine.

No type checker plugins, no build steps, no code generation. Everything works with `ty`, `mypy`, and `pyright` as they exist today.

### 1.3 Non-Goals

- Colnade is **not** a new DataFrame execution engine. It generates and delegates to existing engines.
- Colnade does **not** aim to cover every possible DataFrame operation. It targets the relational core (filter, select, project, aggregate, join, sort, limit, with_columns) and provides escape hatches for engine-specific features.
- Colnade does **not** attempt full type-level schema inference for all transformations. It provides static safety for the common cases and requires explicit schema declarations at transformation boundaries where the type system cannot infer the result.

---

## 2. Prior Art and Lessons Learned

### 2.1 StaticFrame 2

**Approach:** Uses PEP 646 `TypeVarTuple` to encode column types as variadic generic parameters: `Frame[Any, Index[np.str_], np.int_, np.str_, np.float64]`.

**What worked:** Proved that DataFrame type hints can be used for static analysis with Pyright. Immutable data model makes type annotations trustworthy.

**What failed:**

- Column types are **positional, not named**. No way to express "has a column called `age` of type `UInt8`."
- Schema transformations (select, aggregate, join) are not tracked—return types collapse to `Frame[Any, ...]`.
- Only works with Pyright; mypy's `TypeVarTuple` support remains incomplete years later.
- Requires adopting a niche DataFrame engine, which is the primary adoption barrier.

**Lesson:** Don't encode schema as generic type parameters. Don't require a new execution engine.

### 2.2 Pandera (mypy integration)

**Approach:** Define `DataFrameModel` classes; use `DataFrame[Schema]` as a generic type hint. Provides a mypy plugin for basic static checking.

**What worked:** Schema-as-class pattern is intuitive. Wide backend support (Pandas, Polars, Spark, etc.).

**What failed:**

- Static checking is nominal only: it verifies that `DataFrame[A]` is passed where `DataFrame[A]` is expected, but cannot verify that the dataframe's contents actually match after mutation.
- Pandera's own docs state: "since pandas dataframes are mutable objects, there's no way for mypy to know whether a mutated instance has the correct contents."
- The mypy integration is marked "experimental" and produces frequent false positives.
- Column references within function bodies are still unchecked strings.

**Lesson:** Nominal schema types at function boundaries are insufficient. The type checker must also understand column references inside function bodies.

### 2.3 strictly_typed_pandas

**Approach:** `DataSet[Schema]` subclass of `pd.DataFrame` that validates on construction and is immutable.

**What worked:** Immutability makes the type annotation trustworthy—if you have a `DataSet[Schema]`, it genuinely conforms.

**What failed:**

- Any DataFrame operation (`.assign()`, `.iloc`, etc.) returns a plain `pd.DataFrame`, losing type safety.
- Must re-wrap into `DataSet[NewSchema]` after every transformation.
- No column reference checking within function bodies.

**Lesson:** Immutability is the right model, but the entire operation API must return typed results, not drop back to untyped DataFrames.

### 2.4 Patito

**Approach:** Pydantic-style model classes that double as Polars DataFrame schemas. Runtime validation via `.validate()`.

**What worked:** Clean API design. Good integration with Polars. Models serve as single source of truth for data shape.

**What failed:**

- Purely runtime validation. No static type checking of column references.
- `pl.col("misspelled")` passes the type checker and fails at runtime.

**Lesson:** The schema-as-model pattern is good. The missing piece is making column references statically verifiable.

### 2.5 Narwhals

**Approach:** Lightweight compatibility layer providing a Polars-like API across Pandas, Polars, cuDF, PyArrow, DuckDB, PySpark, etc.

**What worked:** Proved that a thin abstraction layer over multiple engines is viable and adoptable. Used by Altair, Plotly, scikit-lego, and many others. Zero-dependency design.

**What failed (for our purposes):**

- No schema typing. Column references are strings. No static safety.
- Solves "write once, run on any engine" but not "catch errors before runtime."

**Lesson:** The cross-engine abstraction model works and is adoptable. Colnade should complement or build upon this pattern rather than compete with it.

### 2.6 TypeVarDict Proposal (python/typing #1387)

**Approach:** Proposed new type variable kind for TypedDict-like record types, with `TD.key`/`TD.value` accessors and a `Map` type operator.

**What would work:** Would enable proper generic DataFrame typing at the language level: `df["a"]` returning `Series[np.int64]` based on the schema.

**What failed:**

- Stalled discussion with no PEP or implementation. Eric Traut (pyright author) expressed skepticism about the approach, suggesting `typeddict_transform` instead.
- PEP 695's new type parameter syntax may have consumed the `**` syntax slot that `TypeVarDict` would need.
- Even if accepted, would take years to land in CPython + all type checkers.

**Lesson:** Don't wait for language-level support. Design around what type checkers can do today.

---

## 3. Architecture

### 3.1 Layer Diagram

```
┌──────────────────────────────────────────────────────┐
│                    User Code                         │
│  (schemas, transforms, pipeline definitions)         │
├──────────────────────────────────────────────────────┤
│              colnade (core library)                │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Schema    │  │  Expression  │  │  DataFrame   │ │
│  │   Layer     │  │    DSL       │  │  Interface   │ │
│  └────────────┘  └──────────────┘  └──────────────┘ │
├──────────────────────────────────────────────────────┤
│                 Backend Adapters                      │
│  ┌────────┐ ┌──────────┐ ┌─────┐ ┌────────────────┐ │
│  │ Polars │ │ Snowpark │ │ Ray │ │ DuckDB / etc.  │ │
│  └────────┘ └──────────┘ └─────┘ └────────────────┘ │
├──────────────────────────────────────────────────────┤
│              Execution Engines                        │
│  (Polars, Snowflake, Ray Data, DuckDB, etc.)         │
└──────────────────────────────────────────────────────┘
```

### 3.2 Package Structure

```
colnade/                     # Core library (zero heavy dependencies)
├── __init__.py
├── schema.py                   # Schema base class, Column descriptors
├── dtypes.py                   # Type definitions (UInt8, Float64, Utf8, etc.)
├── expr.py                     # Typed expression tree
├── dataframe.py                # DataFrame[S], LazyFrame[S], JoinedDataFrame[S, S2], etc.
├── series.py                   # Series[T] generic interface
├── ops.py                      # Aggregation, window, etc.
├── udf.py                      # @udf decorator for typed batch UDFs
├── _protocols.py               # Backend protocols (what adapters must implement)
├── _types.py                   # Internal typing utilities
└── py.typed                    # PEP 561 marker

colnade-polars/              # Polars backend adapter
├── __init__.py
├── adapter.py                  # Expression tree → Polars expressions
├── io.py                       # read_parquet, read_csv, etc.
└── conversion.py               # Arrow boundary handling

colnade-duckdb/              # DuckDB backend adapter (future)
colnade-snowpark/            # Snowpark backend adapter (future)
colnade-ray/                 # Ray Data backend adapter (future)
```

### 3.3 Dependency Policy

- `colnade` core has **zero runtime dependencies** beyond `typing_extensions` (for Python <3.12 compatibility). All type definitions, schema classes, and expression tree nodes are pure Python.
- Backend adapter packages depend on their respective engine (`colnade-polars` depends on `polars`, etc.).
- No build steps, no code generation, no type checker plugins.

---

## 4. Schema Layer

### 4.1 Schema Definition

Schemas are defined as classes extending `Schema`, which itself extends `Protocol`. This enables structural subtyping: any schema that has a superset of another schema's fields is automatically a structural subtype.

```python
from colnade import Schema, UInt64, UInt8, Utf8, Float64, Datetime

class Users(Schema):
    id: UInt64
    name: Utf8
    age: UInt8 | None
    score: Float64
    created_at: Datetime
```

### 4.2 Column Descriptors

Each field on a `Schema` class is not a plain type annotation—it is a **column descriptor**. The `Schema` metaclass replaces each annotation with a `Column[T, S]` descriptor object, where `T` is the column's data type and `S` is the owning schema.

```python
# At runtime, after class creation:
Users.age  # → Column[UInt8 | None, Users]
Users.name  # → Column[Utf8, Users]
```

Column descriptors serve triple duty:

1. **Type-checked attribute access.** `Users.agee` is an `AttributeError` caught by the type checker.
2. **Expression builder.** `Users.age > 18` builds `Expr[Bool]` via `__gt__`.
3. **Column identity.** Used in `.select()`, `.group_by()`, `.as_column()`, etc., replacing string-based column names.

### 4.3 Type Representation for Type Checkers

The type checker sees `Column` as a generic descriptor:

```python
from typing import Generic, TypeVar, Protocol, overload

DType = TypeVar("DType")
SchemaType = TypeVar("SchemaType", bound="Schema")

class Column(Generic[DType, SchemaType]):
    """A typed reference to a named column in a schema."""

    @overload
    def __gt__(self: Column[NumericType, SchemaType], other: int | float) -> Expr[Bool]: ...
    @overload
    def __gt__(self: Column[NumericType, SchemaType], other: Column[NumericType, SchemaType]) -> Expr[Bool]: ...

    def sum(self: Column[NumericType, SchemaType]) -> Agg[NumericType]: ...
    def mean(self: Column[NumericType, SchemaType]) -> Agg[Float64]: ...
    def count(self) -> Agg[UInt32]: ...
    def max(self) -> Agg[DType]: ...
    def min(self) -> Agg[DType]: ...
    def fill_null(self, value: DType) -> Expr[DType]: ...
    def is_null(self) -> Expr[Bool]: ...
    def alias(self, target: Column[DType, Any]) -> AliasedExpr[DType]: ...
    def cast(self, dtype: type[NewDType]) -> Expr[NewDType]: ...

    # String-specific methods (only available on Column[Utf8, S])
    def str_contains(self: Column[Utf8, SchemaType], pattern: str) -> Expr[Bool]: ...
    def str_len(self: Column[Utf8, SchemaType]) -> Expr[UInt32]: ...

    # Datetime-specific methods (only available on Column[Datetime, S])
    def dt_year(self: Column[Datetime, SchemaType]) -> Expr[Int32]: ...
    def dt_month(self: Column[Datetime, SchemaType]) -> Expr[UInt8]: ...

    # Struct-specific methods (only available on Column[Struct[S2], S])
    def field(
        self: Column[Struct[S2], SchemaType],
        col: Column[T, S2],
    ) -> Expr[T]: ...

    # List-specific accessor (only available on Column[List[T], S])
    @property
    def list(self: Column[List[T], SchemaType]) -> ListAccessor[T]: ...
```

The `ListAccessor` provides list-specific operations:

```python
class ListAccessor(Generic[T]):
    def len(self) -> Expr[UInt32]: ...
    def get(self, index: int) -> Expr[T | None]: ...
    def contains(self, value: T) -> Expr[Bool]: ...
    def sum(self: ListAccessor[NumericType]) -> Expr[NumericType | None]: ...
    def mean(self: ListAccessor[NumericType]) -> Expr[Float64 | None]: ...
    def min(self: ListAccessor[NumericType]) -> Expr[NumericType | None]: ...
    def max(self: ListAccessor[NumericType]) -> Expr[NumericType | None]: ...
```

The key insight is that **method availability is type-conditional.** `.sum()` only exists on numeric columns; `.str_contains()` only on `Utf8` columns; `.field()` only on `Struct` columns; `.list` only on `List` columns. The type checker enforces this via `self` type narrowing (a standard Python typing feature). Calling `Users.name.sum()` is a static type error.

### 4.4 Schema Metaclass Behavior

The `Schema` metaclass performs the following at class creation time:

1. **Collects annotations** from the class and all bases (supporting inheritance).
2. **Creates `Column` descriptor objects** for each field, storing the column name, dtype, and owning schema class.
3. **Registers the schema** in an internal registry for runtime validation support.
4. **Generates `__init_subclass__` hooks** so that subclass schemas correctly inherit and extend parent fields.

```python
class SchemaMeta(type(Protocol)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        annotations = {}
        for base in reversed(cls.__mro__):
            annotations.update(getattr(base, '__annotations__', {}))

        cls._columns = {}
        for col_name, col_type in annotations.items():
            if col_name.startswith('_'):
                continue
            descriptor = Column(name=col_name, dtype=col_type, schema=cls)
            setattr(cls, col_name, descriptor)
            cls._columns[col_name] = descriptor

        return cls
```

### 4.5 Schema Inheritance and Composition

Schemas support standard Python class inheritance:

```python
class Users(Schema):
    id: UInt64
    name: Utf8
    age: UInt8
    score: Float64

# Adding columns
class EnrichedUsers(Users):
    normalized_age: Float64
    age_bucket: Utf8

# Subset schema (for projected results)
class UserSummary(Schema):
    name: Utf8
    score: Float64

# Aggregation output
class AgeStats(Schema):
    age: UInt8
    avg_score: Float64
    user_count: UInt32

# Trait-style composition
class HasUserId(Schema):
    user_id: UInt64

class HasTimestamp(Schema):
    created_at: Datetime

class Events(HasUserId, HasTimestamp):
    event_type: Utf8
    payload: Utf8
```

Schema inheritance means:

- `EnrichedUsers` has all `Users` columns plus its own. **One line per added column, not full redeclaration.**
- `Events` composes traits, inheriting `user_id` from `HasUserId` and `created_at` from `HasTimestamp`.
- `EnrichedUsers` is a structural subtype of `Users` at the Schema/Protocol level, meaning it satisfies `bound=Users` constraints in TypeVar bounds.

**DataFrame is invariant.** `DataFrame[EnrichedUsers]` is **not** directly assignable to `DataFrame[Users]` — DataFrame is invariant in its schema parameter. This is intentional: covariant mutable containers are unsound, and even though Colnade DataFrames are immutable, covariance would require `Column[Any, EnrichedUsers]` to be accepted where `Column[Any, Users]` is expected, creating a cascade of contravariance requirements on Column.

Instead, generic utility functions use **bounded TypeVar**, which provides the same ergonomics without variance:

```python
S = TypeVar("S", bound=HasAge)

def filter_adults(df: DataFrame[S]) -> DataFrame[S]:
    return df.filter(HasAge.age >= 18)

# Called with DataFrame[Users] — S resolves to Users, return is DataFrame[Users]
# Called with DataFrame[EnrichedUsers] — S resolves to EnrichedUsers, return is DataFrame[EnrichedUsers]
```

This pattern is strictly more useful than covariance: the return type preserves the full concrete schema, not just the supertype.

**Immutability.** All Colnade DataFrame types are immutable by design. No operation modifies a frame in place — `filter`, `select`, `with_columns`, `sort`, etc. all return new instances. There is no `__setitem__`, `__delitem__`, or `inplace` parameter. This follows Polars' model and is a deliberate departure from pandas' mutable model. Immutability is what makes the schema type annotation trustworthy: if you have a `DataFrame[Users]`, it genuinely conforms to `Users` for its entire lifetime.

### 4.6 Column Source Mapping with `mapped_from`

When a schema represents the output of a join or other cross-schema transformation, columns may need to be mapped from source schemas with different names. The `mapped_from()` field modifier declares this mapping at the schema level:

```python
from colnade import Schema, UInt64, Utf8, Float64, mapped_from

class Users(Schema):
    id: UInt64
    name: Utf8

class Orders(Schema):
    order_id: UInt64
    user_id: UInt64
    amount: Float64

class UserOrders(Schema):
    user_name: Utf8 = mapped_from(Users.name)       # "name" → "user_name"
    order_id: UInt64 = mapped_from(Orders.order_id)  # no rename, explicit source
    amount: Float64 = mapped_from(Orders.amount)
```

`mapped_from` serves two purposes:

1. **Declares the source column** for `cast_schema` to use when resolving columns from a `JoinedDataFrame` (see Section 6.5).
2. **Acts as documentation** — the schema definition is a single source of truth for what data comes from where.

At the type level, `mapped_from(Users.name)` returns a value that the type checker sees as compatible with the field's declared type (`Utf8`). If the source column's type doesn't match the field's declared type, it's a static error:

```python
class Bad(Schema):
    user_name: Float64 = mapped_from(Users.name)  # Static error: Utf8 ≠ Float64
```

`mapped_from` is optional. Columns without it are matched by name during `cast_schema` (the common case when column names don't need renaming).

### 4.7 Null Handling

#### 4.7.1 Nullable Types in Schemas

Nullable columns use Python's native `None` in union syntax:

```python
class Users(Schema):
    id: UInt64           # Non-nullable — must always have a value
    age: UInt8 | None    # Nullable — may be absent
    email: Utf8 | None   # Nullable
```

This is idiomatic Python — the same syntax used in dataclasses, TypedDict, and function signatures. Every Python developer reads `UInt8 | None` as "might not be there" without learning a new concept.

#### 4.7.2 Naming Convention: `None` in Types, `null` in Methods

Schema annotations use `None` (Python convention), but expression methods use `null` (DataFrame/SQL convention):

```python
# Schema: Python idiom
class Users(Schema):
    age: UInt8 | None

# Expressions: DataFrame/SQL idiom
Users.age.is_null()          # not is_none()
Users.age.fill_null(0)       # not fill_none()
Users.age.assert_non_null()  # not assert_not_none()
```

**Rationale:** These are two different contexts with two different established vocabularies. `None` is what Python developers write in type annotations. `is_null` / `fill_null` is what every DataFrame library (Polars, Spark, DuckDB) and SQL uses for missingness operations. Developers already hold both vocabularies — they write `Optional[int]` in Python and `WHERE x IS NOT NULL` in SQL. Forcing one convention into the other's domain (`age: UInt8 | Null` or `age.fill_none(0)`) would feel wrong to whichever audience it borrows from. Using each community's native convention in its own context minimizes friction.

#### 4.7.3 `null` vs. `NaN`

Colnade treats null (absent value) and NaN (IEEE 754 undefined float result) as distinct concepts:

- **`None` / null** — the value is absent. Orthogonal to dtype. Any column type can be nullable.
- **`NaN`** — a valid `Float64` value representing an undefined mathematical result (0/0, sqrt(-1)). Not a null. Not special to the type system. Only exists in float columns.

This follows the model established by Polars, DuckDB, and Arrow, and corrects the conflation that pandas made by using `NaN` as its null sentinel for float columns (which is the root cause of pandas' infamous int→float coercion when nulls appear in integer columns).

```python
class Measurements(Schema):
    sensor_id: UInt64
    reading: Float64           # Non-nullable, but can contain NaN
    temperature: Float64 | None  # Nullable AND can contain NaN

# These are different operations:
Measurements.reading.is_nan()       # Expr[Bool] — "is this an undefined float?"
Measurements.temperature.is_null()  # Expr[Bool] — "is this value absent?"

# Clean up both independently:
Measurements.reading.fill_nan(0.0)           # Expr[Float64] — replace NaN
Measurements.temperature.fill_null(lit(0.0)) # Expr[Float64] — replace null
```

`is_nan()` and `fill_nan()` are only available on float columns — calling them on `UInt8` or `Utf8` is a static type error.

#### 4.7.4 Null Propagation in Expressions

Operations on nullable columns produce nullable results:

```python
class Users(Schema):
    age: UInt8 | None    # nullable
    score: Float64       # non-nullable

Users.age > 18           # Expr[Bool | None] — comparison with nullable input
Users.score > 50.0       # Expr[Bool] — non-nullable, clean
Users.age + Users.score  # Expr[Float64 | None] — null propagates through arithmetic
```

Null-stripping operations remove `None` from the expression type:

```python
Users.age.fill_null(0)              # Expr[UInt8] — no longer nullable
Users.age.assert_non_null()         # Expr[UInt8] — strips null, inserts runtime check
```

These are encoded as overloads: `fill_null` on `Column[T | None, S]` returns `Expr[T]`.

#### 4.7.5 Nullability Enforcement in `cast_schema`

`cast_schema` enforces nullability compatibility at the type level. Narrowing a nullable column to a non-nullable target is a static error:

```python
class UsersClean(Schema):
    age: UInt8       # non-nullable target
    score: Float64

df: DataFrame[Users]  # Users.age is UInt8 | None

# Static error: UInt8 | None is not assignable to UInt8
df.cast_schema(UsersClean)

# Fix 1: strip nulls from the expression type
df.with_columns(
    Users.age.fill_null(0)
).cast_schema(UsersClean)  # OK — fill_null produces Expr[UInt8]

# Fix 2: assert non-null (runtime-validated)
df.with_columns(
    Users.age.assert_non_null()
).cast_schema(UsersClean)  # OK — assert_non_null produces Expr[UInt8]

# Fix 3: drop rows (runtime-validated at cast_schema boundary)
df.drop_nulls(Users.age).cast_schema(UsersClean)
```

Note that `drop_nulls` returns `DataFrame[Users]` (the schema still says `age: UInt8 | None`), so the nullability narrowing happens at `cast_schema` time as a runtime check. The `fill_null` and `assert_non_null` paths are preferable because the expression type itself is non-nullable, making the `cast_schema` call statically sound.

Widening (non-nullable to nullable) is always permitted — `UInt8` is assignable to `UInt8 | None`.

#### 4.7.6 Backend Null Normalization

Backend adapters normalize engine-specific null representations to Colnade's uniform model:

- **Polars adapter:** Direct mapping — Polars already distinguishes `null` from `NaN` and supports nullable integer types natively.
- **Pandas adapter:** Converts columns to pandas nullable dtypes (`pd.UInt8Dtype()`, `pd.Float64Dtype()`, `pd.StringDtype()`) where `pd.NA` represents null and `NaN` remains a float value. This prevents pandas' default behavior of coercing integer columns to `float64` when nulls are present.
- **DuckDB / Arrow adapters:** Direct mapping — both use Arrow's null bitmap, which cleanly separates null from NaN.

**Sentinel values at ingestion.** CSV files and other text sources may contain string representations of missingness (`"NA"`, `""`, `"null"`, `"NaN"`). Backend `read_csv` methods accept a `null_values` parameter to control which strings are parsed as null. Schema validation at the read boundary then catches any remaining nulls in non-nullable columns.

### 4.8 Nested Types: Struct and List

#### 4.8.1 Struct Columns

Struct columns use an existing `Schema` class as their type parameter. This means schemas serve dual duty — as DataFrame-level schemas and as struct field descriptors — which is natural since both are just "a collection of named, typed fields."

```python
class Address(Schema):
    street: Utf8
    city: Utf8
    zip: Utf8

class GeoPoint(Schema):
    lat: Float64
    lng: Float64

class Users(Schema):
    id: UInt64
    name: Utf8
    address: Struct[Address]             # struct column
    location: Struct[GeoPoint] | None    # nullable struct
```

The Schema metaclass handles `Struct[Address]` the same as any other dtype — it creates a `Column[Struct[Address], Users]` descriptor. No special-casing is needed because `Struct[Address]` is simply a parameterized type like `List[Utf8]`.

**Typed struct field access** uses the `.field()` method with a column descriptor from the struct's schema, replacing string-based access:

```python
# Typed field access — no strings:
Users.address.field(Address.city)       # Expr[Utf8]
Users.address.field(Address.zip)        # Expr[Utf8]
Users.location.field(GeoPoint.lat)      # Expr[Float64 | None] — nullable struct propagates

# Chained operations work naturally:
Users.address.field(Address.city).str_contains("New York")  # Expr[Bool]

# Static errors caught:
Users.address.field(Address.city).sum()     # Error: sum() not available on Utf8
Users.address.field(Users.name)             # Error: name is not a field of Address
Users.name.field(Address.city)              # Error: field() not available on Utf8
```

The `.field()` method is only available on `Column[Struct[S2], S]` and accepts `Column[T, S2]` — the struct schema parameter `S2` must match. This means the type checker verifies both that you're accessing a struct column *and* that the field belongs to that struct's schema.

#### 4.8.2 List Columns

List columns are parameterized by their element type:

```python
class Users(Schema):
    id: UInt64
    tags: List[Utf8]                     # list of strings
    scores: List[Float64 | None]         # list of nullable floats
    friends: List[UInt64] | None         # nullable list of non-nullable integers
```

List operations are available via the `.list` accessor, which is only present on `Column[List[T], S]`:

```python
# List operations:
Users.tags.list.len()                    # Expr[UInt32]
Users.tags.list.get(0)                   # Expr[Utf8 | None] — index access may be out of bounds
Users.tags.list.contains("admin")        # Expr[Bool]

# Numeric list aggregations (only on List[NumericType]):
Users.scores.list.sum()                  # Expr[Float64 | None]
Users.scores.list.mean()                 # Expr[Float64 | None]
Users.scores.list.min()                  # Expr[Float64 | None]
Users.scores.list.max()                  # Expr[Float64 | None]

# Static errors:
Users.tags.list.sum()                    # Error: sum() not on ListAccessor[Utf8]
Users.name.list.len()                    # Error: list accessor not available on Utf8
```

#### 4.8.3 Nested Nullability

Nullability composes naturally at each level of nesting:

```python
List[Utf8]                  # non-nullable list of non-nullable strings
List[Utf8 | None]           # non-nullable list of nullable strings
List[Utf8] | None           # nullable list of non-nullable strings
List[Utf8 | None] | None    # nullable list of nullable strings

Struct[Address]              # non-nullable struct
Struct[Address] | None       # nullable struct (the whole struct may be absent)
# Individual struct field nullability is defined by Address's field annotations
```

Outer nullability propagates through access:

```python
class Users(Schema):
    location: Struct[GeoPoint] | None   # nullable struct

Users.location.field(GeoPoint.lat)      # Expr[Float64 | None]
# Even though GeoPoint.lat is non-nullable Float64, accessing it through
# a nullable struct produces a nullable result.
```

#### 4.8.4 Phase 1 Scope and Phase 2 Deferral

Phase 1 includes `Struct[S]`, `List[T]`, `.field()`, `.list.*` accessors, and nested nullability. This ensures the core type machinery (Schema metaclass, Column descriptors, expression tree, backend adapters) handles parameterized dtypes from day one.

Deferred to Phase 2:

- **Deeply nested access:** `Struct[Struct[...]]` and `List[Struct[...]]` compositions. These compose in theory but require extensive edge-case testing.
- **List aggregation in group_by context:** `explode`, `flatten`, and similar reshaping operations.
- **Struct construction expressions:** Building a struct column from individual scalar columns.
- **Nested type operations in window functions.**

---

## 5. Expression DSL

### 5.1 Expression Tree

All operations on `Column` descriptors produce typed expression tree nodes rather than immediately executing. This allows backend adapters to translate the entire expression to engine-native operations.

```python
class Expr(Generic[DType]):
    """Base class for all expression tree nodes."""
    pass

class ColumnRef(Expr[DType]):
    """Reference to a schema column."""
    column: Column[DType, Any]

class BinOp(Expr[DType]):
    """Binary operation (arithmetic, comparison, logical)."""
    left: Expr
    right: Expr
    op: str  # "+", "-", ">", "==", "&", "|", etc.

class UnaryOp(Expr[DType]):
    """Unary operation (negation, not, is_null, etc.)."""
    operand: Expr
    op: str

class Literal(Expr[DType]):
    """A literal value."""
    value: Any

class FunctionCall(Expr[DType]):
    """Named function application (str_contains, dt_year, cast, etc.)."""
    name: str
    args: tuple[Expr, ...]
    kwargs: dict[str, Any]

class Agg(Expr[DType]):
    """Aggregation expression (sum, mean, count, etc.)."""
    source: Expr
    agg_type: str  # "sum", "mean", "count", "max", "min", etc.

class StructFieldAccess(Expr[DType]):
    """Access a field within a struct column."""
    struct_expr: Expr[Struct[Any]]
    field: Column[DType, Any]

class ListOp(Expr[DType]):
    """Operation on a list column (len, get, contains, sum, etc.)."""
    list_expr: Expr[List[Any]]
    op: str  # "len", "get", "contains", "sum", "mean", etc.
    args: tuple[Any, ...]

class AliasedExpr(Expr[DType]):
    """Expression with an output column binding."""
    expr: Expr[DType]
    target: Column[DType, Any]
```

### 5.2 Expression Building Examples

```python
# Comparison — nullable input propagates
Users.age > 18              # Expr[Bool | None] (age is UInt8 | None)
Users.score > 50.0          # Expr[Bool] (score is non-nullable Float64)

# Arithmetic — null propagates through operations
Users.age + Users.score     # Expr[Float64 | None] (type promotion + null propagation)

# Chained — both nullable
(Users.age > 18) & (Users.score > 50.0)  # Expr[Bool | None]

# Aggregation — returns Agg[Float64]
Users.score.mean()

# Aliased aggregation — returns AliasedExpr[Float64]
Users.score.mean().as_column(AgeStats.avg_score)

# String operation — only valid on Utf8 columns
Users.name.str_contains("Smith")  # Expr[Bool]

# Type error: .sum() not available on Utf8
Users.name.sum()  # Static type error

# Type error: .str_contains() not available on numeric
Users.age.str_contains("x")  # Static type error

# Struct field access — typed, no strings
Users.address.field(Address.city)                        # Expr[Utf8]
Users.address.field(Address.city).str_contains("York")   # Expr[Bool]

# List operations — via .list accessor
Users.tags.list.len()            # Expr[UInt32]
Users.tags.list.get(0)           # Expr[Utf8 | None]
Users.scores.list.sum()          # Expr[Float64 | None]

# Type error: .field() not available on non-struct column
Users.name.field(Address.city)   # Static type error

# Type error: wrong struct schema for .field()
Users.address.field(GeoPoint.lat)  # Static type error: GeoPoint ≠ Address
```

### 5.3 Type Promotion Rules

Arithmetic between column types follows standard numeric promotion:

| Left       | Right      | Result     |
|------------|------------|------------|
| UInt8      | UInt8      | UInt8      |
| UInt8      | Int32      | Int32      |
| Int64      | Float64    | Float64    |
| Any integer| float literal | Float64 |
| T          | T          | T          |

These are encoded as `@overload` signatures on the `Column` and `Expr` operator methods.

---

## 6. DataFrame Interface

### 6.1 Core Types: DataFrame, LazyFrame, and JoinedDataFrame

`DataFrame[S]` represents materialized data in memory. `LazyFrame[S]` represents an unevaluated query plan. `JoinedDataFrame[S, S2]` (and `JoinedLazyFrame[S, S2]`) represent the result of joining two schemas, accepting column references from either input.

```python
S = TypeVar("S", bound=Schema)
S2 = TypeVar("S2", bound=Schema)
S3 = TypeVar("S3", bound=Schema)

class DataFrame(Generic[S]):
    """A typed, materialized DataFrame parameterized by a Schema."""

    def filter(self, predicate: Expr[Bool]) -> DataFrame[S]: ...
    def sort(self, *columns: Column[Any, S], descending: bool = False) -> DataFrame[S]: ...
    def limit(self, n: int) -> DataFrame[S]: ...
    def head(self, n: int = 5) -> DataFrame[S]: ...
    def tail(self, n: int = 5) -> DataFrame[S]: ...
    def sample(self, n: int) -> DataFrame[S]: ...

    # Select: overloaded for arities 1–10 to constrain column inputs to S.
    # Return type is DataFrame[Any] — output schema requires cast_schema.
    # See Section 6.3 for details.
    @overload
    def select(self, c1: Column[Any, S], /) -> DataFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any, S], c2: Column[Any, S], /) -> DataFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any, S], c2: Column[Any, S], c3: Column[Any, S], /) -> DataFrame[Any]: ...
    # ... overloads up to 10 columns
    def select(self, *columns: Column[Any, S]) -> DataFrame[Any]: ...

    def with_columns(self, *exprs: AliasedExpr | Expr) -> DataFrame[S]: ...

    def group_by(self, *keys: Column[Any, S]) -> GroupBy[S]: ...

    def join(
        self,
        other: DataFrame[S2],
        on: JoinCondition,
        how: Literal["inner", "left", "outer", "cross"] = "inner",
    ) -> JoinedDataFrame[S, S2]: ...

    def unique(self, *columns: Column[Any, S]) -> DataFrame[S]: ...
    def drop_nulls(self, *columns: Column[Any, S]) -> DataFrame[S]: ...

    def cast_schema(
        self,
        schema: type[S2],
        mapping: dict[Column, Column] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> DataFrame[S2]: ...

    # Conversion
    def lazy(self) -> LazyFrame[S]: ...
    def untyped(self) -> UntypedDataFrame: ...

    # Boundary methods
    def to_batches(self, batch_size: int | None = None) -> Iterator[ArrowBatch[S]]: ...
    def validate(self) -> DataFrame[S]: ...


class LazyFrame(Generic[S]):
    """A typed, lazy query plan parameterized by a Schema."""

    def filter(self, predicate: Expr[Bool]) -> LazyFrame[S]: ...
    def sort(self, *columns: Column[Any, S], descending: bool = False) -> LazyFrame[S]: ...
    def limit(self, n: int) -> LazyFrame[S]: ...

    @overload
    def select(self, c1: Column[Any, S], /) -> LazyFrame[Any]: ...
    @overload
    def select(self, c1: Column[Any, S], c2: Column[Any, S], /) -> LazyFrame[Any]: ...
    # ... overloads up to 10 columns
    def select(self, *columns: Column[Any, S]) -> LazyFrame[Any]: ...

    def with_columns(self, *exprs: AliasedExpr | Expr) -> LazyFrame[S]: ...

    def group_by(self, *keys: Column[Any, S]) -> LazyGroupBy[S]: ...

    def join(
        self,
        other: LazyFrame[S2],
        on: JoinCondition,
        how: Literal["inner", "left", "outer", "cross"] = "inner",
    ) -> JoinedLazyFrame[S, S2]: ...

    def unique(self, *columns: Column[Any, S]) -> LazyFrame[S]: ...
    def drop_nulls(self, *columns: Column[Any, S]) -> LazyFrame[S]: ...

    def cast_schema(
        self,
        schema: type[S2],
        mapping: dict[Column, Column] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> LazyFrame[S2]: ...

    # Materialization
    def collect(self) -> DataFrame[S]: ...

    # Conversion
    def untyped(self) -> UntypedLazyFrame: ...


class JoinedDataFrame(Generic[S, S2]):
    """Result of joining two DataFrames. Accepts columns from either input schema."""

    def filter(self, predicate: Expr[Bool]) -> JoinedDataFrame[S, S2]: ...

    @overload
    def select(self, c1: Column[Any, S] | Column[Any, S2], /) -> DataFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any, S] | Column[Any, S2],
        c2: Column[Any, S] | Column[Any, S2], /
    ) -> DataFrame[Any]: ...
    # ... overloads up to 10 columns
    def select(self, *columns: Column[Any, S] | Column[Any, S2]) -> DataFrame[Any]: ...

    def sort(self, *columns: Column[Any, S] | Column[Any, S2]) -> JoinedDataFrame[S, S2]: ...
    def unique(self, *columns: Column[Any, S] | Column[Any, S2]) -> JoinedDataFrame[S, S2]: ...
    def drop_nulls(self, *columns: Column[Any, S] | Column[Any, S2]) -> JoinedDataFrame[S, S2]: ...
    def limit(self, n: int) -> JoinedDataFrame[S, S2]: ...

    def with_columns(self, *exprs: AliasedExpr | Expr) -> JoinedDataFrame[S, S2]: ...

    def cast_schema(
        self,
        schema: type[S3],
        mapping: dict[Column, Column] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> DataFrame[S3]: ...

    # Conversion
    def lazy(self) -> JoinedLazyFrame[S, S2]: ...
    def untyped(self) -> UntypedDataFrame: ...


class JoinedLazyFrame(Generic[S, S2]):
    """Lazy equivalent of JoinedDataFrame."""

    def filter(self, predicate: Expr[Bool]) -> JoinedLazyFrame[S, S2]: ...

    @overload
    def select(self, c1: Column[Any, S] | Column[Any, S2], /) -> LazyFrame[Any]: ...
    @overload
    def select(
        self,
        c1: Column[Any, S] | Column[Any, S2],
        c2: Column[Any, S] | Column[Any, S2], /
    ) -> LazyFrame[Any]: ...
    # ... overloads up to 10 columns
    def select(self, *columns: Column[Any, S] | Column[Any, S2]) -> LazyFrame[Any]: ...

    def sort(self, *columns: Column[Any, S] | Column[Any, S2]) -> JoinedLazyFrame[S, S2]: ...
    def unique(self, *columns: Column[Any, S] | Column[Any, S2]) -> JoinedLazyFrame[S, S2]: ...
    def drop_nulls(self, *columns: Column[Any, S] | Column[Any, S2]) -> JoinedLazyFrame[S, S2]: ...
    def limit(self, n: int) -> JoinedLazyFrame[S, S2]: ...

    def with_columns(self, *exprs: AliasedExpr | Expr) -> JoinedLazyFrame[S, S2]: ...

    def cast_schema(
        self,
        schema: type[S3],
        mapping: dict[Column, Column] | None = None,
        extra: Literal["drop", "forbid"] = "drop",
    ) -> LazyFrame[S3]: ...

    def collect(self) -> JoinedDataFrame[S, S2]: ...
    def untyped(self) -> UntypedLazyFrame: ...
```

**JoinCondition** constrains the join predicate to compare columns that exist in their respective schemas:

```python
class JoinCondition:
    """A join predicate produced by comparing columns from two different schemas.
    Created by the == operator between Column descriptors from different schemas."""
    left: Column[Any, Any]
    right: Column[Any, Any]

# Created by:
Users.id == Orders.user_id  # → JoinCondition (when schemas differ)
Users.age == Users.score     # → Expr[Bool] (same schema, normal comparison)
```

The `==` operator on `Column` is overloaded: when both columns share the same schema parameter, it returns `Expr[Bool]` (a filter predicate). When the schema parameters differ, it returns `JoinCondition` (a join specification). This distinction allows `join(on=...)` to accept only cross-schema comparisons while `filter(...)` accepts only same-schema expressions.

**Multi-way joins** are handled by flattening between each join:

```python
user_orders = users.join(orders, on=Users.id == Orders.user_id).cast_schema(UserOrders)
full = user_orders.join(products, on=UserOrders.product_id == Products.id).cast_schema(FullReport)
```

Each intermediate join result is `cast_schema`'d to a named schema before the next join. This forces intermediates to be named and documented, avoids recursive type nesting, and keeps each join's type signature clean.

The key distinction between the three types: `DataFrame[S]` is a single-schema frame, `JoinedDataFrame[S, S2]` is a two-schema frame, and you cannot pass a `JoinedDataFrame` where a `DataFrame` is expected — you must `cast_schema` first. This is correct: a joined result genuinely has different semantics (two namespaces, potential column collisions) that should be resolved explicitly before further processing.

Operations that only make sense on materialized data (`.head()`, `.tail()`, `.sample()`, `.to_batches()`) exist only on `DataFrame` and `JoinedDataFrame`, not their lazy counterparts.

### 6.2 Schema-Preserving Operations

Operations that do not change the schema return `DataFrame[S]` (or `LazyFrame[S]`) directly:

```python
df: DataFrame[Users]

# All of these return DataFrame[Users]:
df.filter(Users.age > 18)
df.sort(Users.name)
df.limit(100)
df.unique(Users.id)
df.drop_nulls(Users.age)

# Same for lazy:
lf: LazyFrame[Users]
lf.filter(Users.age > 18)  # LazyFrame[Users]
```

`with_columns` that overwrites an existing column **with the same type** also preserves the schema:

```python
df.with_columns(Users.age + 1)  # DataFrame[Users] — age is still UInt8
```

**`with_columns` semantics in detail:**

`with_columns` always returns `DataFrame[S]` (preserving the current schema type). This is an optimistic "trust me" zone — the type system assumes the schema is preserved, and `cast_schema` is where truth is enforced. The three cases:

- **Overwrite existing column, same type:** Sound. The schema annotation remains truthful. This is the common case (e.g., normalizing a column, filling nulls).
- **Overwrite existing column, different type:** The schema annotation becomes a lie (says old type, data has new type). This cannot be caught statically because `with_columns` doesn't know at the type level which column an expression targets. It will surface as a runtime type mismatch at the next `cast_schema` call.
- **Add a new column not in S:** The column exists in the underlying data but is invisible to the type system. It will be dropped at the next `cast_schema(extra="drop")` (the default) or included if the target schema declares it. This is the standard pattern for building up to a richer schema:

```python
class EnrichedUsers(Users):
    risk_score: Float64

def add_risk(df: DataFrame[Users]) -> DataFrame[EnrichedUsers]:
    return df.with_columns(
        (Users.age * 0.1 + Users.score * 0.9).as_column(EnrichedUsers.risk_score)
    ).cast_schema(EnrichedUsers)
    # with_columns adds the data; cast_schema makes it visible to the type system
```

### 6.3 Schema-Transforming Operations

Operations that change the schema require an explicit output schema. Rather than trying to compute output schemas at the type level (which Python's type system cannot do), the developer declares what comes out.

**Select/Project:**

`select` is overloaded for arities 1–10. The overloads constrain each column argument to `Column[Any, S]`, ensuring the type checker verifies that all selected columns belong to the input schema. The return type is `DataFrame[Any]`, requiring `cast_schema` to bind the result to a named output schema:

```python
class UserNames(Schema):
    name: Utf8
    score: Float64

def get_names(df: DataFrame[Users]) -> DataFrame[UserNames]:
    return df.select(Users.name, Users.score).cast_schema(UserNames)

# Static error: Orders.amount doesn't belong to Users
df.select(Users.name, Orders.amount)  # type error at overload resolution
```

The overloads provide **input validation** (all columns belong to `S`) but not **output inference** (the result schema must be declared). This is a deliberate trade-off: output inference would require type-level record computation that Python doesn't support, while input validation is achievable with standard overloads and catches the most common bugs (referencing columns from the wrong schema).

**Aggregation:**

```python
class AgeStats(Schema):
    age: UInt8
    avg_score: Float64
    user_count: UInt32

def summarize(df: DataFrame[Users]) -> DataFrame[AgeStats]:
    return (
        df.group_by(Users.age)
        .agg(
            Users.score.mean().as_column(AgeStats.avg_score),
            Users.id.count().as_column(AgeStats.user_count),
        )
        .cast_schema(AgeStats)
    )
```

The `.as_column(Target.col)` method binds an expression's output to a specific column in the target schema. The types must be compatible: `Users.score.mean()` produces `Agg[Float64]`, and `AgeStats.avg_score` is `Column[Float64, AgeStats]`. If the types don't match, it's a static error.

**Adding columns** follows the `with_columns` + `cast_schema` pattern described in Section 6.2.

### 6.4 `cast_schema` Method

`cast_schema` is the explicit boundary marker where the developer asserts that the DataFrame now conforms to a new schema. At runtime, this performs a validation check (column names and types match). At the type level, it changes the schema parameter:

```python
def cast_schema(
    self,
    schema: type[S2],
    mapping: dict[Column, Column] | None = None,
    extra: Literal["drop", "forbid"] = "drop",
) -> DataFrame[S2]: ...
```

**Parameters:**

- **`schema`**: The target schema class.
- **`mapping`** (optional): Explicit column mapping for one-off renames. Takes precedence over `mapped_from` for any columns it covers.
- **`extra`**: How to handle source columns not declared in the target schema.
  - `"drop"` (default): Silently discard extra columns, projecting to only the target schema's columns. This matches SQL SELECT semantics and is the right default for pipelines where intermediate columns are common.
  - `"forbid"`: Raise `SchemaError` if the source has columns not in the target schema. Useful for strict validation at pipeline boundaries where unexpected columns indicate a bug.

**Name matching (default):** When `mapping` is `None`, `cast_schema` matches output schema columns to source columns by name. If the output schema has `mapped_from()` annotations (see Section 4.6), those are used instead.

```python
# Simple case — column names match between source and target:
df.select(Users.name, Users.score).cast_schema(UserNames)

# mapped_from case — schema declares source mapping:
class UserOrders(Schema):
    user_name: Utf8 = mapped_from(Users.name)  # rename: name → user_name
    amount: Float64 = mapped_from(Orders.amount)

joined_df.cast_schema(UserOrders)  # mapping resolved from schema definition
```

**Explicit mapping (one-off):** For ad-hoc remapping without declaring `mapped_from` on the schema:

```python
joined_df.cast_schema(UserOrders, mapping={
    UserOrders.user_name: Users.name,
})
```

This is analogous to `DataSet[Schema](df)` in `strictly_typed_pandas`, but as a method call rather than a constructor. `cast_schema` (along with generic passthrough via TypeVar) is the only way to change a DataFrame's schema type.

### 6.5 Joins and `JoinedDataFrame`

Joins return `JoinedDataFrame[S, S2]` — a distinct type where columns from both input schemas are accessible via their original descriptors:

```python
class Users(Schema):
    id: UInt64
    name: Utf8
    age: UInt8

class Orders(Schema):
    order_id: UInt64
    user_id: UInt64
    amount: Float64

joined = users_df.join(orders_df, on=Users.id == Orders.user_id)
# joined: JoinedDataFrame[Users, Orders]

# Access columns via their original schema — no flat namespace collision:
joined.filter(Users.age > 18)                   # ✅ Users.age still valid
joined.filter(Orders.amount > 100.0)            # ✅ Orders.amount still valid
joined.select(Users.name, Orders.amount, Orders.order_id)  # ✅ mixed sources
```

`JoinedDataFrame` is a separate type from `DataFrame` because its methods accept `Column[Any, S] | Column[Any, S2]` (columns from either input schema), whereas `DataFrame[S]` methods only accept `Column[Any, S]`. This distinction is what makes the joined column access type-safe — the type checker verifies that each column belongs to one of the two joined schemas.

**No collision problem.** If both schemas have a column with the same name (e.g., both have `id`), you disambiguate by which descriptor you use: `Users.id` vs `Orders.id`. The backend adapter handles internal disambiguation (e.g., Polars suffixes, SQL qualified names). The user never sees the internal naming scheme — they always work through schema descriptors.

**Materializing a joined result.** `JoinedDataFrame` is an intermediate type for continued operations. Before writing output, passing to another system, or calling functions that expect a specific schema, you `cast_schema` to a flat output schema:

```python
class UserOrders(Schema):
    user_name: Utf8 = mapped_from(Users.name)
    order_id: UInt64 = mapped_from(Orders.order_id)
    amount: Float64 = mapped_from(Orders.amount)

def join_user_orders(
    users: DataFrame[Users],
    orders: DataFrame[Orders],
) -> DataFrame[UserOrders]:
    return (
        users.join(orders, on=Users.id == Orders.user_id)
        .cast_schema(UserOrders)  # mapped_from resolves column sources
    )
```

Three resolution strategies in `cast_schema` on a `JoinedDataFrame`, in order of precedence:

1. **Explicit `mapping` parameter** — if provided, used directly.
2. **`mapped_from()` on the target schema** — used if declared on the field.
3. **Name matching** — if the column name exists unambiguously in the joined result, matched automatically. This covers the common case where column names don't collide and don't need renaming.

For the simple case with no collisions or renames, `cast_schema` just works by name:

```python
# If UserOrders fields had the same names as their sources,
# no mapped_from would be needed:
class SimpleUserOrders(Schema):
    name: Utf8       # matches Users.name unambiguously
    order_id: UInt64  # matches Orders.order_id unambiguously
    amount: Float64   # matches Orders.amount unambiguously

users_df.join(orders_df, on=...).cast_schema(SimpleUserOrders)  # just works
```

### 6.6 GroupBy

```python
class GroupBy(Generic[S]):
    def agg(self, *exprs: AliasedExpr) -> DataFrame[Any]:
        """
        Perform aggregation. Returns untyped DataFrame that must be
        cast to an explicit output schema via .cast_schema().
        """
        ...
```

### 6.7 The Untyped Escape Hatch

For exploratory work, prototyping, or operations not yet covered by the typed API:

```python
class UntypedDataFrame:
    """A DataFrame with no schema parameter. String-based column access."""

    def select(self, *columns: str) -> UntypedDataFrame: ...
    def filter(self, expr: Any) -> UntypedDataFrame: ...
    def to_typed(self, schema: type[S]) -> DataFrame[S]: ...
    # ... full Polars-like API with string column references
```

Usage:

```python
# Drop into untyped mode for complex or unsupported operations
result = (
    df.untyped()
    .with_columns(some_complex_operation())
    .to_typed(OutputSchema)
)
```

`to_typed()` performs runtime validation. This is the "TypeScript `any`" equivalent — it exists for pragmatism but can be linted against in CI (e.g., a custom lint rule that flags `.untyped()` calls).

---

## 7. Generic Utility Functions and Schema Polymorphism

This is where the design differentiates from all prior art. The goal is to write functions that are generic over schemas, preserving full type information through transformations, without type checker plugins or code generation.

### 7.1 Passthrough Transforms (S → S)

The most common pattern. A function that takes a DataFrame with some schema, performs operations that don't change the schema, and returns the same schema:

```python
S = TypeVar("S", bound=Schema)

def drop_null_rows(df: DataFrame[S]) -> DataFrame[S]:
    """Works on any schema. Preserves full schema in return type."""
    return df.drop_nulls()
```

When called:

```python
users_df: DataFrame[Users]
result = drop_null_rows(users_df)  # result: DataFrame[Users]
```

The type checker resolves `S = Users` and the return type is `DataFrame[Users]` — full schema preserved.

### 7.2 Constrained Transforms (requires specific columns)

Functions that need specific columns use Protocol-based bounds:

```python
class HasAge(Schema):
    age: UInt8

S = TypeVar("S", bound=HasAge)

def filter_adults(df: DataFrame[S]) -> DataFrame[S]:
    """Works on any schema that has an age: UInt8 column."""
    return df.filter(HasAge.age >= 18)
```

When called:

```python
users_df: DataFrame[Users]
result = filter_adults(users_df)  # result: DataFrame[Users]

# Users structurally satisfies HasAge (it has age: UInt8), so S resolves to Users.
# The full Users schema is preserved in the return type.
```

If called with a schema that lacks `age`:

```python
class Products(Schema):
    id: UInt64
    name: Utf8
    price: Float64

products_df: DataFrame[Products]
filter_adults(products_df)  # Static type error: Products doesn't satisfy HasAge
```

### 7.3 Column-Parameterized Transforms

For functions that are generic over which column they operate on:

```python
from colnade import NumericType

N = TypeVar("N", bound=NumericType)

def normalize_column(
    df: DataFrame[S],
    col: Column[N, S],
) -> DataFrame[S]:
    """Normalize any numeric column in any schema to [0, 1]."""
    return df.with_columns(
        (col - col.min()) / (col.max() - col.min())
    )
```

Usage:

```python
normalize_column(users_df, Users.age)     # OK: age is numeric, in Users
normalize_column(users_df, Users.name)    # Static error: name is Utf8, not numeric
normalize_column(users_df, Orders.amount) # Static error: amount is not in Users
```

The constraint `Column[N, S]` means "a column of numeric type `N` belonging to schema `S`." The type checker unifies `S` from the `df` argument and verifies that the column belongs to the same schema.

### 7.4 Multi-Column Constraints

```python
class HasScoreAndAge(Schema):
    age: UInt8
    score: Float64

S = TypeVar("S", bound=HasScoreAndAge)

def age_weighted_score(df: DataFrame[S]) -> DataFrame[S]:
    """Requires both age and score columns."""
    return df.with_columns(
        (HasScoreAndAge.score * (HasScoreAndAge.age / 100.0))
    )
```

### 7.5 What Cannot Be Expressed

The following pattern **cannot** be expressed without a type checker plugin:

```python
# HYPOTHETICAL — NOT POSSIBLE without plugin:
def add_timestamp(df: DataFrame[S]) -> DataFrame[S & {timestamp: Datetime}]:
    ...
```

"Take any schema S, return S plus a new column" requires type-level record combination, which Python's type system doesn't support. The practical alternatives are:

1. **Declare the output schema** using inheritance (one line per added column):

```python
class UsersWithTimestamp(Users):
    timestamp: Datetime

def add_timestamp(df: DataFrame[Users]) -> DataFrame[UsersWithTimestamp]:
    ...
```

2. **Accept the concrete types** rather than being generic over S:

```python
# If truly needed as generic, accept that the caller must declare the output.
```

In practice, "add a column to an unknown schema" is rare in production code. Most column addition happens at known pipeline stages with known schemas.

---

## 8. Backend Adapters

### 8.1 Backend Protocol

Each backend adapter implements the following protocol:

```python
from typing import Protocol, Iterator
from colnade.expr import Expr
from colnade.schema import Schema

class BackendProtocol(Protocol):
    """What a backend adapter must implement."""

    def execute_filter(self, source: Any, predicate: Expr[Bool]) -> Any: ...
    def execute_select(self, source: Any, columns: list[Column]) -> Any: ...
    def execute_with_columns(self, source: Any, exprs: list[Expr]) -> Any: ...
    def execute_group_by_agg(
        self, source: Any, keys: list[Column], aggs: list[AliasedExpr]
    ) -> Any: ...
    def execute_join(
        self, left: Any, right: Any, condition: JoinCondition, how: str
    ) -> Any: ...
    def execute_sort(self, source: Any, columns: list[Column], descending: bool) -> Any: ...
    def execute_limit(self, source: Any, n: int) -> Any: ...

    def read_parquet(self, path: str, schema: type[Schema]) -> Any: ...
    def read_csv(self, path: str, schema: type[Schema]) -> Any: ...
    def to_arrow_batches(self, source: Any, batch_size: int | None) -> Iterator[Any]: ...
    def from_arrow_batches(self, batches: Iterator[Any], schema: type[Schema]) -> Any: ...
    def validate_schema(self, source: Any, schema: type[Schema]) -> None: ...

    def translate_expr(self, expr: Expr) -> Any:
        """Convert a Colnade expression tree to engine-native expression."""
        ...
```

### 8.2 Polars Adapter (Reference Implementation)

The Polars adapter translates Colnade expression trees into Polars expressions:

```python
# colnade_polars/adapter.py

import polars as pl
from colnade.expr import (
    Expr, ColumnRef, BinOp, Literal, Agg, FunctionCall, UnaryOp
)

class PolarsAdapter:
    def translate_expr(self, expr: Expr) -> pl.Expr:
        match expr:
            case ColumnRef(column=col):
                return pl.col(col.name)
            case BinOp(left=l, right=r, op="+"):
                return self.translate_expr(l) + self.translate_expr(r)
            case BinOp(left=l, right=r, op=">"):
                return self.translate_expr(l) > self.translate_expr(r)
            case BinOp(left=l, right=r, op="&"):
                return self.translate_expr(l) & self.translate_expr(r)
            case Literal(value=v):
                return pl.lit(v)
            case Agg(source=s, agg_type="mean"):
                return self.translate_expr(s).mean()
            case Agg(source=s, agg_type="sum"):
                return self.translate_expr(s).sum()
            case Agg(source=s, agg_type="count"):
                return self.translate_expr(s).count()
            case FunctionCall(name="str_contains", args=(s, pattern)):
                return self.translate_expr(s).str.contains(pattern.value)
            case FunctionCall(name="fill_null", args=(s, val)):
                return self.translate_expr(s).fill_null(self.translate_expr(val))
            case FunctionCall(name="cast", args=(s,), kwargs={"dtype": dtype}):
                return self.translate_expr(s).cast(self._map_dtype(dtype))
            # ... etc.

    def execute_filter(self, source: pl.DataFrame, predicate: Expr[Bool]) -> pl.DataFrame:
        return source.filter(self.translate_expr(predicate))

    def execute_select(self, source: pl.DataFrame, columns: list[Column]) -> pl.DataFrame:
        return source.select([pl.col(c.name) for c in columns])

    def read_parquet(self, path: str, schema: type[Schema]) -> pl.DataFrame:
        df = pl.read_parquet(path)
        self.validate_schema(df, schema)
        return df

    def validate_schema(self, df: pl.DataFrame, schema: type[Schema]) -> None:
        expected_cols = schema._columns
        for name, col in expected_cols.items():
            if name not in df.columns:
                raise SchemaError(f"Missing column: {name}")
            actual_dtype = df.schema[name]
            expected_dtype = self._map_dtype(col.dtype)
            if actual_dtype != expected_dtype:
                raise SchemaError(
                    f"Column {name}: expected {expected_dtype}, got {actual_dtype}"
                )
```

### 8.3 Expression AST Node Coverage

The expression AST that adapters must handle:

| Node Type           | Examples                                           | Polars Equivalent           |
|---------------------|----------------------------------------------------|-----------------------------|
| `ColumnRef`         | `Users.age`                                        | `pl.col("age")`            |
| `Literal`           | `42`, `"hello"`, `3.14`                            | `pl.lit(42)`               |
| `BinOp`             | `+`, `-`, `*`, `/`, `>`, `<`, `>=`, `<=`, `==`, `!=`, `&`, `\|` | Native operators |
| `UnaryOp`           | `-`, `~` (not), `is_null`, `is_not_null`           | `.is_null()`, etc.          |
| `Agg`               | `sum`, `mean`, `count`, `min`, `max`, `first`, `last`, `std`, `var` | `.sum()`, `.mean()`, etc. |
| `FunctionCall`      | `str_contains`, `str_len`, `dt_year`, `dt_month`, `cast`, `fill_null`, `abs`, `round` | Method calls |
| `StructFieldAccess` | `Users.address.field(Address.city)`                | `.struct.field("city")`     |
| `ListOp`            | `.list.len()`, `.list.get(0)`, `.list.sum()`       | `.list.len()`, etc.         |
| `AliasedExpr`       | `.as_column(Target.col)`                           | `.alias("col")`            |
| `SortExpr`          | `col.desc()`, `col.asc()`                          | `.sort(descending=True)`    |
| `WindowExpr`        | `col.over(partition_by=...)`                       | `.over(...)`                |

### 8.4 Adding a New Backend

To add support for a new engine (e.g., DuckDB):

1. Create a new package `colnade-duckdb`.
2. Implement the `BackendProtocol` interface.
3. Implement `translate_expr` via pattern matching on the expression AST (approximately 20–30 cases).
4. Implement I/O methods (read_parquet, read_csv, to_arrow_batches, etc.).
5. Implement `validate_schema` using the engine's native schema introspection.

The expression AST is intentionally small and maps closely to the relational algebra, making translation straightforward for any SQL or DataFrame engine.

---

## 9. Cross-Framework Boundaries

### 9.1 Arrow as Lingua Franca

When data moves between engines, Apache Arrow is the serialization format. Every serious DataFrame engine speaks Arrow natively.

```python
class ArrowBatch(Generic[S]):
    """A typed wrapper around a pyarrow.RecordBatch."""
    _batch: pa.RecordBatch
    _schema: type[S]

    def to_pyarrow(self) -> pa.RecordBatch:
        return self._batch

    @classmethod
    def from_pyarrow(cls, batch: pa.RecordBatch, schema: type[S]) -> ArrowBatch[S]:
        """Validate and wrap an Arrow batch."""
        _validate_arrow_schema(batch.schema, schema)
        return cls(_batch=batch, _schema=schema)
```

### 9.2 Boundary Transitions

```python
# Snowflake → Ray transition
snow_df: DataFrame[Users] = snow_backend.table("users", schema=Users)
batches: Iterator[ArrowBatch[Users]] = snow_df.to_batches()

ray_ds: RayDataset[Users] = ray_backend.from_batches(batches, schema=Users)
```

The schema type parameter flows through the boundary. `to_batches()` on `DataFrame[Users]` produces `Iterator[ArrowBatch[Users]]`. `from_batches` accepts `Iterator[ArrowBatch[Users]]` and produces `RayDataset[Users]`. If you accidentally pass `ArrowBatch[Orders]` to a function expecting `ArrowBatch[Users]`, the type checker catches it.

Runtime validation occurs at the boundary (checking that Arrow column names and types match the schema). This is cheap and desirable when data crosses system boundaries.

### 9.3 Typed Batch UDFs

For distributed engines that process data in batches (Ray Data, Snowpark UDFs):

```python
from colnade import udf, DataFrame, Schema

class ScoredUsers(Users):
    risk_score: Float64

@udf(input_schema=Users, output_schema=ScoredUsers)
def compute_risk(batch: DataFrame[Users]) -> DataFrame[ScoredUsers]:
    return batch.with_columns(
        (Users.age * 0.1 + Users.score * 0.9).as_column(ScoredUsers.risk_score)
    ).cast_schema(ScoredUsers)
```

The `@udf` decorator:

1. Validates at decoration time that the function's type annotations match `input_schema` and `output_schema`.
2. At runtime, wraps incoming engine-native batches in `DataFrame[InputSchema]` with validation.
3. Calls the function.
4. Validates the output against `OutputSchema`.
5. Returns the engine-native batch.

For Ray:

```python
ray_ds: RayDataset[Users] = ...
scored: RayDataset[ScoredUsers] = ray_ds.map_batches(compute_risk)
```

The `map_batches` signature is overloaded so that applying a `udf(input=A, output=B)` to `RayDataset[A]` yields `RayDataset[B]`.

---

## 10. Type System Coverage Matrix

Summary of what the type checker can and cannot verify:

| Operation                                    | Static Check | Mechanism                          |
|----------------------------------------------|:------------:|------------------------------------|
| Column reference exists in schema            | ✅           | Attribute access on Protocol class |
| Column reference has correct type            | ✅           | `Column[DType, Schema]` generic    |
| Method availability by dtype (e.g., `.sum()` on numeric) | ✅ | Self-type narrowing on `Column` |
| Filter/sort/limit preserves schema           | ✅           | Returns `DataFrame[S]`            |
| Same-type column overwrite preserves schema  | ✅           | Returns `DataFrame[S]`            |
| Function schema passthrough                  | ✅           | Bounded `TypeVar`                 |
| Schema structural subtyping                  | ✅           | `Protocol` structural matching     |
| Select/sort/group_by columns belong to schema | ✅          | `Column[Any, S]` constraint       |
| Joined select/sort accepts both input schemas | ✅          | `Column[Any, S] \| Column[Any, S2]` |
| Expression type correctness                  | ✅           | `Expr[DType]` generics + overloads|
| Join condition compares cross-schema columns | ✅           | `JoinCondition` vs `Expr[Bool]` overload |
| UDF input/output schema match                | ✅           | Decorator type constraints         |
| Cross-framework boundary schema match        | ✅           | `ArrowBatch[S]` generics          |
| Lazy vs. eager distinction                   | ✅           | `LazyFrame[S]` vs `DataFrame[S]`  |
| JoinedDataFrame vs DataFrame distinction     | ✅           | Separate types, must `cast_schema` |
| `mapped_from` source type matches field type | ✅           | `mapped_from` return type check    |
| Null propagation through expressions         | ✅           | `Expr[T \| None]` via overloads    |
| `fill_null`/`assert_non_null` strips nullability | ✅       | Overloads return `Expr[T]`         |
| Nullability narrowing in `cast_schema`       | ✅           | `T \| None` not assignable to `T`  |
| `is_nan`/`fill_nan` only on float columns    | ✅           | Method only on `Float32`/`Float64` |
| Struct field access type-safe                | ✅           | `.field()` constrained to `Column[Struct[S2], S]` |
| Struct field belongs to correct struct schema | ✅          | `.field(Column[T, S2])` checks `S2` matches |
| List operations only on list columns         | ✅           | `.list` accessor on `Column[List[T], S]` only |
| List element type flows through operations   | ✅           | `.list.get()` returns `Expr[T \| None]`   |
| Wrong-schema column in filter/with_columns   | ❌           | `Expr[Bool]` erases source schema  |
| Select/drop infers output schema             | ❌           | Requires explicit output schema    |
| Aggregation infers output schema             | ❌           | Requires explicit output schema    |
| Join infers combined flat schema             | ❌           | Requires `cast_schema` to named schema |
| "Add column to generic S" (`S & {col: T}`)  | ❌           | Requires explicit subclass of S    |

---

## 11. Error Messages

Good error messages are critical for adoption. The type checker should produce clear, actionable errors:

### 11.1 Misspelled Column

```python
df.filter(Users.agee > 18)
# ty error: Schema "Users" has no attribute "agee". Did you mean "age"?
```

This is a standard attribute error on a class—type checkers already produce good messages for this.

### 11.2 Wrong Type Operation

```python
Users.name.sum()
# ty error: "Column[Utf8, Users]" has no attribute "sum".
#           "sum" is only available on numeric column types.
```

### 11.3 Column from Wrong Schema (Partial)

Expressions erase their source schema — `Orders.amount > 100` and `Users.age > 18` both produce `Expr[Bool]`. Once built, the type checker cannot distinguish them. This means `filter` and `with_columns` do not check that expressions reference columns from the correct schema:

```python
def process(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.filter(Orders.amount > 100)
# No static error — Expr[Bool] is Expr[Bool] regardless of source.
# Fails at runtime when the backend can't find "amount" in the Users frame.
```

However, the type checker **does** catch wrong-schema columns in methods that accept `Column[Any, S]` directly:

```python
df.select(Orders.amount)  # Static error: Column[Float64, Orders] ≠ Column[Any, Users]
df.sort(Orders.amount)     # Static error
df.group_by(Orders.amount) # Static error
```

And on `JoinedDataFrame[S, S2]`, both `Column[Any, S]` and `Column[Any, S2]` are accepted, so columns from unrelated schemas are still caught:

```python
joined: JoinedDataFrame[Users, Orders]
joined.select(Products.price)  # Static error: Products is neither Users nor Orders
```

This is a conscious gap in the type system coverage. The column *existence* check (catching typos like `Users.agee`) and column *type* check (catching `Users.name.sum()`) cover the most common bugs. The "valid column from wrong schema in filter/with_columns" case is less common and always fails clearly at runtime.

### 11.4 Schema Mismatch at Boundary

```python
def process(df: DataFrame[Users]) -> DataFrame[Orders]:
    return df  # type error: DataFrame[Users] is not assignable to DataFrame[Orders]
```

### 11.5 Incompatible as_column Binding

```python
Users.name.sum()  # type error: sum() not available on Utf8

Users.age.mean().as_column(AgeStats.user_count)
# type error: Agg[Float64] is not assignable to Column[UInt32, AgeStats]
```

### 11.6 Nullability Mismatch

```python
class UsersClean(Schema):
    age: UInt8  # non-nullable

df: DataFrame[Users]  # Users.age is UInt8 | None
df.cast_schema(UsersClean)
# type error: Column "age" is UInt8 | None in source but UInt8 in target.
#             Use fill_null() or assert_non_null() to handle nullability.
```

---

## 12. Runtime Behavior

### 12.1 Schema Validation

At runtime, schema validation checks:

1. All expected columns are present.
2. Column data types match (using a mapping from Colnade dtypes to engine-native dtypes).
3. Non-nullable columns have no null values (optional, configurable).

```python
class SchemaError(Exception):
    """Raised when data does not conform to the declared schema."""
    missing_columns: list[str]
    extra_columns: list[str]
    type_mismatches: dict[str, tuple[str, str]]  # col → (expected, actual)
    null_violations: list[str]
```

### 12.2 When Validation Runs

- On `cast_schema()` — always.
- On backend `read_*()` methods — always.
- On `ArrowBatch.from_pyarrow()` — always.
- On `@udf` decorated function entry and exit — always.
- On `DataFrame.__init__()` when constructing from raw data — always.
- On `filter()`, `sort()`, etc. (schema-preserving ops) — **never** (no validation needed; schema is preserved by construction).

### 12.3 Lazy vs. Eager

`DataFrame[S]` and `LazyFrame[S]` are distinct types reflecting the fundamental difference between materialized data and query plans.

**Eager (`DataFrame[S]`):** Operations execute immediately. `cast_schema()` validation runs on the materialized data. All operations including `.head()`, `.sample()`, and `.to_batches()` are available.

**Lazy (`LazyFrame[S]`):** Operations build a query plan. No data moves until `.collect()` is called, at which point the backend can optimize the full plan (predicate pushdown, projection pushdown, join reordering, etc.). `cast_schema()` validation is deferred to `.collect()` time.

```python
# Eager
df: DataFrame[Users] = polars_backend.read_parquet("data.parquet", schema=Users)
result: DataFrame[Users] = df.filter(Users.age > 18)  # Executes immediately

# Lazy
lf: LazyFrame[Users] = polars_backend.scan_parquet("data.parquet", schema=Users)
plan: LazyFrame[Users] = lf.filter(Users.age > 18)    # Builds plan, no execution
result: DataFrame[Users] = plan.collect()              # Executes optimized plan

# Type error: LazyFrame is not DataFrame
def process(df: DataFrame[Users]) -> DataFrame[Users]: ...
process(lf)         # Static type error
process(lf.collect())  # OK
```

This explicitness prevents a common class of performance bugs where intermediate results are accidentally materialized in a lazy pipeline. It also mirrors how backends actually work — Polars has `DataFrame` vs `LazyFrame`, DuckDB and Spark are inherently lazy, Ray Data is lazy-ish — making the adapter mapping natural.

---

## 13. Project Phasing

### Phase 1: Core Library + Polars Adapter

**Scope:**

- `colnade` core: Schema, Column descriptors, Expression DSL, `DataFrame[S]` and `LazyFrame[S]` interfaces, `JoinedDataFrame[S, S2]` and `JoinedLazyFrame[S, S2]` types, `JoinCondition`, `mapped_from()`.
- Nested types: `Struct[S]`, `List[T]`, typed `.field()` access, `.list.*` accessors, nested nullability.
- `colnade-polars`: Polars backend adapter with full expression translation, eager and lazy support.
- Runtime schema validation (including nested type validation).
- Static type checking verified against `ty`, `mypy`, and `pyright`.

**Deliverables:**

- Published packages on PyPI.
- Comprehensive test suite covering all typed operations including struct/list.
- Type checker test suite: a set of `.py` files with expected type errors, run against all three type checkers in CI.
- Documentation with examples.

**Success criteria:** A user can define schemas (including struct and list columns), write fully type-checked DataFrame pipelines (eager and lazy), and run them on Polars with zero string-based column references in their code.

### Phase 2: Arrow Boundaries + Additional Backends

**Scope:**

- `ArrowBatch[S]` typed boundary protocol.
- `@udf` decorator for typed batch UDFs.
- DuckDB backend adapter.
- Advanced nested type operations: deeply nested access (`Struct[Struct[...]]`, `List[Struct[...]]`), list aggregation in group_by context (`explode`, `flatten`), struct construction expressions, nested type window functions.

**Deliverables:**

- Cross-engine data transfer with typed boundaries.
- `colnade-duckdb` package.
- UDF documentation and examples.

### Phase 3: Distributed Engine Adapters

**Scope:**

- `colnade-ray`: Ray Data adapter with typed `map_batches`.
- `colnade-snowpark`: Snowpark adapter (expression → SQL translation).
- End-to-end multi-engine pipeline examples.

### Phase 4: Type Checker Plugin (Optional Enhancement)

**Scope:**

- A `ty` / `pyright` plugin that can evaluate `Extend[S, col=Type]`, `Omit[S, col]`, `Pick[S, col1, col2]` at type-checking time.
- Enables fully generic "add column to any schema" patterns.
- Optional—all functionality from Phases 1–3 works without it.

---

## 14. Testing Strategy

### 14.1 Runtime Tests

Standard pytest suite covering:

- Schema creation and column descriptor behavior.
- Expression tree construction and type correctness.
- Backend adapter translation (expression → Polars/DuckDB/etc.).
- Schema validation (positive and negative cases).
- End-to-end pipeline execution.
- Cross-framework boundary transfer.
- UDF decorator behavior.

### 14.2 Static Type Checking Tests

A dedicated test suite of `.py` files that are run through `ty`, `mypy`, and `pyright` in CI. Each file contains annotated expected errors:

```python
# tests/typing/test_column_access.py

from colnade import Schema, UInt8, Utf8, DataFrame

class Users(Schema):
    age: UInt8
    name: Utf8

df: DataFrame[Users]

# Should pass:
Users.age > 18
Users.name.str_contains("Smith")
df.filter(Users.age > 18)

# Should error:
Users.agee  # type: ignore[attr-error]  # EXPECTED ERROR: no attribute "agee"
Users.name.sum()  # type: ignore[attr-error]  # EXPECTED ERROR: no sum on Utf8
```

These tests are run in CI with:

```bash
ty check tests/typing/ --expected-errors
mypy tests/typing/ --strict
pyright tests/typing/
```

### 14.3 Backend Adapter Tests

Each backend adapter has integration tests that verify:

- Expression translation produces correct engine-native expressions.
- End-to-end query execution produces correct results.
- Schema validation catches all categories of mismatch.
- Arrow serialization/deserialization preserves schema.

---

## 15. API Reference Summary

### 15.1 Core Imports

```python
from colnade import (
    # Schema definition
    Schema,

    # Data types
    Bool, UInt8, UInt16, UInt32, UInt64,
    Int8, Int16, Int32, Int64,
    Float32, Float64,
    Utf8, Binary,
    Date, Time, Datetime, Duration,
    List, Struct,

    # Type categories (for method constraints)
    NumericType, TemporalType, StringType,

    # Core classes
    DataFrame, LazyFrame, JoinedDataFrame, JoinedLazyFrame,
    Series, Column, Expr, Agg,
    JoinCondition,

    # Schema utilities
    mapped_from,

    # UDF decorator
    udf,

    # Boundary types
    ArrowBatch,
)
```

### 15.2 Schema Definition

```python
class MySchema(Schema):
    column_name: DType
    nullable_column: DType | None
```

### 15.3 DataFrame Operations

```python
df: DataFrame[S]

# Schema-preserving (return DataFrame[S]):
df.filter(expr: Expr[Bool])
df.sort(*columns: Column, descending: bool = False)
df.limit(n: int)
df.head(n: int = 5)
df.tail(n: int = 5)
df.unique(*columns: Column)
df.drop_nulls(*columns: Column)
df.with_columns(*exprs: Expr)  # preserves S optimistically; see §6.2

# Schema-transforming (return DataFrame[Any], use .cast_schema()):
df.select(*columns: Column)
df.group_by(*keys: Column).agg(*exprs: AliasedExpr)
df.cast_schema(NewSchema, extra="drop"|"forbid")

# Joins (return JoinedDataFrame[S, S2]):
df.join(other, on=JoinCondition, how=how)

# JoinedDataFrame[S, S2] — accepts Column[Any, S] | Column[Any, S2]:
joined.filter(expr: Expr[Bool])
joined.select(*columns)  # → DataFrame[Any]
joined.cast_schema(FlatSchema)  # → DataFrame[FlatSchema]

# Conversion:
df.lazy() -> LazyFrame[S]
df.untyped() -> UntypedDataFrame
lf.collect() -> DataFrame[S]
```

### 15.4 Column / Expression Operations

```python
col: Column[DType, S]

# Comparison (→ Expr[Bool]):
col > value, col < value, col >= value, col <= value, col == value, col != value

# Arithmetic (→ Expr[ResultType]):
col + other, col - other, col * other, col / other, col % other

# Logical (→ Expr[Bool]):
expr & expr, expr | expr, ~expr

# Aggregation (→ Agg[ResultType]):
col.sum(), col.mean(), col.min(), col.max(), col.count()
col.std(), col.var(), col.first(), col.last()
col.n_unique()

# Null handling:
col.is_null() → Expr[Bool]
col.is_not_null() → Expr[Bool]
col.fill_null(value) → Expr[DType]  # strips None from nullable columns
col.assert_non_null() → Expr[DType]  # strips None, inserts runtime check

# NaN handling (Float32/Float64 only):
col.is_nan() → Expr[Bool]
col.fill_nan(value) → Expr[DType]

# String (Utf8 only):
col.str_contains(pattern) → Expr[Bool]
col.str_starts_with(prefix) → Expr[Bool]
col.str_ends_with(suffix) → Expr[Bool]
col.str_len() → Expr[UInt32]
col.str_to_lowercase() → Expr[Utf8]
col.str_to_uppercase() → Expr[Utf8]
col.str_strip() → Expr[Utf8]
col.str_replace(pattern, replacement) → Expr[Utf8]

# Temporal (Datetime only):
col.dt_year() → Expr[Int32]
col.dt_month() → Expr[UInt8]
col.dt_day() → Expr[UInt8]
col.dt_hour() → Expr[UInt8]
col.dt_minute() → Expr[UInt8]
col.dt_second() → Expr[UInt8]
col.dt_truncate(interval) → Expr[Datetime]

# Struct (Struct[S] only):
col.field(Column[T, S]) → Expr[T]  # typed field access

# List (List[T] only):
col.list.len() → Expr[UInt32]
col.list.get(index) → Expr[T | None]
col.list.contains(value) → Expr[Bool]
col.list.sum() → Expr[T | None]        # numeric lists only
col.list.mean() → Expr[Float64 | None]  # numeric lists only
col.list.min() → Expr[T | None]        # numeric lists only
col.list.max() → Expr[T | None]        # numeric lists only

# General:
col.cast(NewDType) → Expr[NewDType]
col.alias(TargetColumn) → AliasedExpr[DType]
col.as_column(TargetColumn) → AliasedExpr[DType]  # alias for .alias()
col.over(*partition_by: Column) → Expr[DType]  # window function
col.desc() → SortExpr
col.asc() → SortExpr
```

### 15.5 Backend Usage

```python
# Polars
import colnade_polars as backend

df = backend.read_parquet("data.parquet", schema=Users)
df = backend.read_csv("data.csv", schema=Users)
backend.write_parquet(df, "output.parquet")

# DuckDB
import colnade_duckdb as backend

df = backend.query("SELECT * FROM 'data.parquet'", schema=Users)
df = backend.table("my_table", schema=Users)
```

---

## 16. Design Decisions (Resolved)

### 16.1 `select` Uses Overloads for Input Validation (Resolved)

`select` is overloaded for arities 1–10 to constrain column arguments to `Column[Any, S]`. This statically ensures that all selected columns belong to the input schema. The return type is `DataFrame[Any]` (or `LazyFrame[Any]`), requiring `cast_schema` to bind to a named output schema.

**Rationale:** Output schema inference would require type-level record computation (e.g., TypeScript's `Pick<T, K>`), which Python's type system doesn't support. Overloads up to arity 10 cover the vast majority of real-world `select` calls. Going higher (e.g., 100) was considered but rejected due to stub bloat and type checker performance degradation for diminishing returns. The overloads' primary value is catching "column from wrong schema" bugs, not inferring output shape.

### 16.2 Joins Return `JoinedDataFrame[S, S2]` as a Distinct Type (Resolved)

Joins return `JoinedDataFrame[S, S2]` (not `DataFrame[Joined[S, S2]]`) — a separate type whose methods accept `Column[Any, S] | Column[Any, S2]`, allowing columns from either input schema. `JoinedDataFrame` cannot be passed where `DataFrame` is expected; you must `cast_schema` to a flat output schema first.

Materializing a `JoinedDataFrame` into a flat schema uses `cast_schema` with three resolution strategies (in precedence order): explicit `mapping` parameter, `mapped_from()` annotations on the target schema, or automatic name matching.

**Rationale:** The original design (`DataFrame[Joined[S, S2]]`) was cleaner conceptually but didn't type-check — `Column[Utf8, Users]` is not assignable to `Column[Any, Joined[Users, Orders]]`, so every column-accepting method on the joined frame would fail at the type level. Making `JoinedDataFrame` a distinct type with `Column[Any, S] | Column[Any, S2]` in its method signatures solves this cleanly. The requirement to `cast_schema` before further processing is correct: a joined result genuinely has different semantics (two namespaces, potential collisions) that should be resolved explicitly. Multi-way joins are handled by `cast_schema`'ing between each join to a named intermediate schema.

### 16.3 `LazyFrame[S]` Is Distinct from `DataFrame[S]` (Resolved)

`LazyFrame[S]` and `DataFrame[S]` are separate types sharing nearly identical operation signatures. `LazyFrame` has `.collect() → DataFrame[S]` and `DataFrame` has `.lazy() → LazyFrame[S]` for conversion.

**Rationale:** Being explicit about evaluation mode prevents accidental materialization (a common performance bug in lazy pipelines), mirrors how backends actually work (Polars, DuckDB, Spark all distinguish), and allows the type system to enforce that operations like `.head()`, `.sample()`, and `.to_batches()` are only available on materialized frames. There is no meaningful downside — the additional type is a small cost for significant safety.

### 16.4 `DataFrame[S]` Is Backend-Agnostic (Resolved)

`DataFrame[Users]` is the same type whether backed by Polars, DuckDB, or Spark. The type system does not distinguish backends. This means that a Polars-backed `DataFrame[Users]` and a DuckDB-backed `DataFrame[Users]` are type-compatible — the type checker would not catch an attempt to join them across backends.

**Alternatives considered:**

- **Second type parameter:** `DataFrame[Users, PolarsBackend]`. Catches cross-backend mixing statically, but threads a second `TypeVar` through every type annotation, generic function, and return type. Significant ergonomic cost.
- **Backend-specific subtypes:** `PolarsDataFrame[Users]`, `DuckDBLazyFrame[Users]`. Catches cross-backend mixing but fractures the unified API — generic functions would need unions or a shared protocol, losing the clean single `DataFrame[S]` interface.

**Rationale for current design:** The vast majority of pipelines use a single backend throughout. Cross-engine transfer is already an explicit act (serialize to Arrow, deserialize on the other side), not something you'd do accidentally. And cross-backend operations (e.g., joining a Polars frame with a DuckDB frame) fail immediately at runtime with a clear error, not silently.

**Migration path if needed:** If multi-backend pipelines in Phase 3 reveal this to be a real pain point, backend-specific subtypes can be introduced non-breakingly — `PolarsDataFrame[S]` as a subclass of `DataFrame[S]`, so existing code that says `DataFrame[S]` continues to work. Functions that need to enforce single-backend usage can narrow to the subtype.

### 16.5 Nested Types Included in Phase 1 (Resolved)

`Struct[S]` and `List[T]` are included in Phase 1 scope. Schemas serve dual duty as DataFrame-level schemas and struct field descriptors. `.field()` provides typed struct access; `.list.*` provides typed list operations. See Section 4.8 for full design.

**Rationale:** Including nested types in Phase 1 stress-tests the core abstractions — if the Schema metaclass, Column descriptors, and expression tree handle parameterized dtypes correctly, the foundation is proven. Deferring to Phase 2 would risk building Phase 1 machinery that assumes DType is always a simple non-parameterized type, creating painful retrofitting. Advanced nested operations (deep nesting, explode, struct construction) are deferred to Phase 2.

### 16.6 Custom Backend Protocol, Not Narwhals Dependency (Resolved)

Colnade uses its own `BackendProtocol` for backend adapters rather than building on top of Narwhals.

**Alternatives considered:**

- **Colnade on top of Narwhals:** Translate Colnade expression AST → Narwhals API calls → engine-native calls. Two translation layers with no benefit.
- **Hybrid:** Use Narwhals for multi-engine translation, add typed schema layer independently.

**Rationale:** The abstraction layers are at different levels. Narwhals abstracts over DataFrame APIs (wraps existing DataFrames, forwards method calls). Colnade abstracts over expression trees (builds an AST, lowers to engine-native calls). Layering Colnade on Narwhals would mean fighting Narwhals' abstraction rather than leveraging it, since we'd still need per-engine code for schema validation, null normalization, `JoinedDataFrame` disambiguation, and lazy plan construction. The expression AST is intentionally small (~20–30 pattern match cases per backend), so the marginal implementation cost of custom adapters is low.

**Narwhals interop (not dependency):** Users with Narwhals-wrapped DataFrames can convert them into Colnade via adapter utility functions:

```python
tf_df = colnade_polars.from_narwhals(nw_df, schema=Users)
```

This is a conversion utility on the adapter, not an architectural dependency on the core.

### 16.7 `DataFrame` Is Invariant and Immutable (Resolved)

`DataFrame[S]` is invariant in its schema parameter: `DataFrame[EnrichedUsers]` is **not** assignable to `DataFrame[Users]`, even though `EnrichedUsers` is a structural subtype of `Users`.

All DataFrame types are immutable by design. No operation modifies a frame in place — `filter`, `select`, `with_columns`, `sort`, etc. all return new instances. There is no `__setitem__`, `__delitem__`, or `inplace` parameter.

**Alternatives considered:**

- **Covariant DataFrame:** Would allow `DataFrame[EnrichedUsers]` → `DataFrame[Users]`, but requires `Column[Any, EnrichedUsers]` to be accepted where `Column[Any, Users]` is expected — creating contravariance requirements on Column that cascade through the type system.

**Rationale:** Invariance is simpler and avoids unsound type interactions. The ergonomic loss is minimal because bounded TypeVar provides the same utility: `S = TypeVar("S", bound=HasAge)` lets generic functions accept `DataFrame[Users]` and return `DataFrame[Users]` (not `DataFrame[HasAge]`), preserving the full concrete schema. This is strictly more useful than covariance. Immutability is what makes the schema type annotation trustworthy — if you have a `DataFrame[Users]`, it genuinely conforms to `Users` for its entire lifetime.

### 16.8 Extra Columns Policy: `cast_schema(extra="drop"|"forbid")` (Resolved)

`cast_schema` accepts an `extra` parameter controlling how source columns not declared in the target schema are handled:

- `"drop"` (default): Silently discard extra columns, projecting to only the target schema's columns. Matches SQL SELECT semantics.
- `"forbid"`: Raise `SchemaError` if extra columns are present. Useful for strict validation at pipeline boundaries.

**Rationale:** Pipelines commonly produce intermediate columns that shouldn't appear in the output. Making `"drop"` the default follows the principle of least surprise for DataFrame workflows. The `"forbid"` option covers strict-validation use cases (similar to Pydantic's `model_config = ConfigDict(extra="forbid")`). A third option `"allow"` (keep extra columns invisible to the type system) was not included because retaining untyped columns undermines the library's core purpose.

### 16.9 `filter` and `with_columns` Do Not Check Expression Schema (Resolved)

`filter` accepts `Expr[Bool]` and `with_columns` accepts `AliasedExpr | Expr`. Because expression trees erase their source schema (e.g., `Orders.amount > 100` and `Users.age > 18` both produce `Expr[Bool]`), these methods cannot verify that expressions reference columns from the correct schema.

**Rationale:** Adding a schema parameter to `Expr` (e.g., `Expr[Bool, Users]`) was considered but creates cascading complexity — what is the schema parameter of `(Users.age > 18) & (Orders.amount > 100)`? The gap is acceptable because: (1) column *existence* is always caught at attribute access time (`Users.agee` → type error), (2) column *type* mismatches are caught (`Users.name.sum()` → type error), (3) methods that accept `Column[Any, S]` directly (select, sort, group_by) do check schema, and (4) wrong-schema columns in filter/with_columns fail clearly at runtime. The most common bugs are covered; the uncovered case (valid column from wrong schema) is relatively rare.

---

## 17. Naming and Packaging

**Name:** `colnade` (from "colonnade" — a row of columns)

**Website:** colnade.com

**Package names:**

- `colnade` — core library
- `colnade-polars` — Polars adapter
- `colnade-duckdb` — DuckDB adapter
- `colnade-ray` — Ray Data adapter
- `colnade-snowpark` — Snowpark adapter

**Python version support:** 3.10+ (for `match` statements and `X | Y` union syntax). 3.12+ recommended (for PEP 695 type parameter syntax). `typing_extensions` used for backcompat where needed.

**License:** MIT

---

## Appendix A: Full Worked Example

```python
# schemas.py
from colnade import Schema, UInt64, UInt32, UInt8, Utf8, Float64, Datetime

class RawEvents(Schema):
    event_id: UInt64
    user_id: UInt64
    event_type: Utf8
    timestamp: Datetime
    payload: Utf8 | None

class Users(Schema):
    user_id: UInt64
    name: Utf8
    age: UInt8
    signup_date: Datetime

class UserFeatures(Schema):
    user_id: UInt64
    name: Utf8
    event_count: UInt32
    last_active: Datetime

class ScoredUsers(UserFeatures):
    risk_score: Float64


# transforms.py
from colnade import DataFrame, udf
from typing import TypeVar
from .schemas import RawEvents, Users, UserFeatures, ScoredUsers

def build_features(
    events: DataFrame[RawEvents],
    users: DataFrame[Users],
) -> DataFrame[UserFeatures]:
    event_agg = (
        events
        .group_by(RawEvents.user_id)
        .agg(
            RawEvents.event_id.count().as_column(UserFeatures.event_count),
            RawEvents.timestamp.max().as_column(UserFeatures.last_active),
        )
        .cast_schema(UserFeatures)  # Validates at runtime
    )
    return event_agg

# Generic utility — works on any schema with user_id
class HasUserId(Schema):
    user_id: UInt64

S = TypeVar("S", bound=HasUserId)

def filter_active_users(
    df: DataFrame[S],
    since: Datetime,
) -> DataFrame[S]:
    """Filter to users who have been active since the given date.
    Works on any schema with a user_id column. Preserves full schema."""
    return df.filter(HasUserId.user_id.is_not_null())

@udf(input_schema=UserFeatures, output_schema=ScoredUsers)
def compute_risk_score(batch: DataFrame[UserFeatures]) -> DataFrame[ScoredUsers]:
    return (
        batch.with_columns(
            (UserFeatures.event_count.cast(Float64) * 0.7
             + UserFeatures.last_active.dt_day().cast(Float64) * 0.3
            ).as_column(ScoredUsers.risk_score)
        )
        .cast_schema(ScoredUsers)
    )


# pipeline.py
import colnade_polars as backend

# Read data
events = backend.read_parquet("events.parquet", schema=RawEvents)
users = backend.read_parquet("users.parquet", schema=Users)

# Transform
features = build_features(events, users)

# Apply ML scoring via UDF
scored = compute_risk_score(features)

# Write output
backend.write_parquet(scored, "scored_users.parquet")
```

Every column reference in this example is statically verified. Every schema transition is explicit. The type checker catches misspelled columns, type mismatches, and schema violations without running the code.
