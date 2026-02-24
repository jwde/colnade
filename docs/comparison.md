# Comparison with Existing Solutions

## Feature Matrix

| Feature | Colnade | Pandera | StaticFrame | Patito | Narwhals |
|---------|---------|---------|-------------|--------|----------|
| Column refs checked statically | Named attrs | No | Positional types¹ | No | No |
| Schema preserved through ops | Through ops² | At boundaries³ | No | No | No |
| Works with existing engines | Polars, Pandas, Dask | Pandas, Polars, others | Own engine | Polars only | Many engines |
| No plugins or code gen | Yes | Requires mypy plugin | Yes | Yes | Yes |
| Generic utility functions | Yes | No | No | No | No |
| Struct/List typed access | Yes | No | No | No | No |
| Lazy execution support | Yes | No | No | No | Yes |
| Value-level constraints | `Field()` | `Check` | No | Pydantic validators | No |
| Maturity / ecosystem | New (v0.6) | Mature, large community | Mature | Small | Growing fast |
| Engine breadth | 3 backends | 4+ backends | Own engine | 1 backend | 6+ backends |
| select/group_by output typing | `DataFrame[Any]`⁴ | Decorator-checked | Positional types | No | No |

¹ StaticFrame encodes column *types* positionally via `TypeVarTuple`, but column *names* are not part of the type signature.

² Schema-preserving ops (filter, sort, with_columns) retain `DataFrame[S]`. Schema-transforming ops (select, group_by) return `DataFrame[Any]` — `cast_schema()` is needed to bind to the output schema and acts as a runtime trust boundary.

³ Pandera's `@check_types` decorator validates schemas at function entry/exit, but column references within function bodies remain unchecked strings (`pa.Column("age")`).

⁴ `cast_schema()` re-binds at runtime. A type checker plugin could theoretically infer output schemas, but Colnade intentionally avoids plugin coupling.

## Detailed Comparisons

### Pandera

Pandera provides runtime schema validation with a mypy plugin for basic static checking. Its `@check_types` decorator validates that `DataFrame[InputSchema]` is passed where expected and that outputs match `DataFrame[OutputSchema]`, providing nominal schema tracking at function boundaries. However, column references within function bodies are unchecked strings — `pa.Column("misspelled")` passes the type checker.

**Colnade's approach:** Column references are class attributes (`Users.age` not `"age"`), so misspellings are caught at lint time. Schema types track through operations, not just at function boundaries. The tradeoff is that schema-transforming operations (select, group_by) erase the schema — `cast_schema()` is needed to re-bind, which Pandera's decorator approach avoids for simple input→output schemas.

### StaticFrame

StaticFrame uses PEP 646 `TypeVarTuple` to encode column types as variadic generic parameters. This gives static type checking of column *types* (you know the frame has an `int` column, a `str` column, etc.), but types are positional, not named — there's no way to express "has a column called `age` of type `UInt8`." It requires adopting StaticFrame's own DataFrame engine rather than working with Polars/Pandas.

**Colnade's approach:** Named columns with full type information. Works with existing engines. The tradeoff is that Colnade requires a separate schema class definition, while StaticFrame infers types from data.

### Patito

Patito provides Pydantic-style model classes that serve as Polars DataFrame schemas, with runtime validation including Pydantic's built-in validators for value constraints. Clean API design, and its model-based approach is similar to Colnade's schema pattern.

**Colnade's approach:** Static checking of all column references, not just runtime validation. Same schema-as-model pattern with similar value constraints (`Field()` vs Pydantic validators), but with compile-time safety for column references. Patito is Polars-only; Colnade supports multiple backends.

### Narwhals

Narwhals is a lightweight compatibility layer providing a Polars-like API across multiple backends (Polars, Pandas, cuDF, Modin, etc.). It solves "write once, run on any engine" but provides no schema typing or static safety.

**Colnade's approach:** Full static type safety for column references and schema transformations. Narwhals has broader engine support. The two libraries solve different problems and could potentially complement each other.

## What Colnade does not do

For transparency, here are things Colnade's type system cannot catch statically:

- **Schema-transforming output types** — `select()` and `group_by().agg()` return `DataFrame[Any]`; `cast_schema()` binds to the output schema at runtime. A type checker plugin could theoretically infer the output schema, but this would couple Colnade to specific type checkers.
- **Cross-schema expression misuse** — `df.filter(Orders.amount > 100)` on a `DataFrame[Users]` is not caught statically (column descriptors lack a schema type parameter). Enable runtime validation to catch this.
- **Literal value types** — `Users.age + "hello"` is not caught (Python lacks type-level functions to map `Column[UInt8]` → operator overloads that only accept `int`).

See [Type Checker Integration](user-guide/type-checking.md) for the full list of what is and isn't checked.
