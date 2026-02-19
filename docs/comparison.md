# Comparison with Existing Solutions

## Feature Matrix

| Feature | Colnade | Pandera | StaticFrame | Patito | Narwhals |
|---------|---------|---------|-------------|--------|----------|
| Column refs checked statically | :white_check_mark: | :x: | :x: | :x: | :x: |
| Schema preserved through ops | :white_check_mark: | Nominal only | :x: | :x: | :x: |
| Works with existing engines | :white_check_mark: | :white_check_mark: | :x: | Polars only | :white_check_mark: |
| No plugins or code gen | :white_check_mark: | :x: (mypy plugin) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Generic utility functions | :white_check_mark: | :x: | :x: | :x: | :x: |
| Struct/List typed access | :white_check_mark: | :x: | :x: | :x: | :x: |
| Lazy execution support | :white_check_mark: | :x: | :x: | :x: | :white_check_mark: |
| Value-level constraints | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: | :x: |

## Detailed Comparisons

### Pandera

Pandera provides runtime schema validation with a mypy plugin for basic static checking. However, its static checking is nominal only — it verifies that `DataFrame[A]` is passed where `DataFrame[A]` is expected, but cannot verify that the dataframe's contents match after transformation. Column references within function bodies remain unchecked strings.

**Colnade's advantage:** Column references are class attributes (`Users.age` not `"age"`), so misspellings are caught at lint time. Schema types track through all operations, not just at function boundaries.

### StaticFrame

StaticFrame uses PEP 646 `TypeVarTuple` to encode column types as variadic generic parameters. Column types are positional, not named — there's no way to express "has a column called `age` of type `UInt8`." Schema transformations aren't tracked, and it requires adopting a niche DataFrame engine.

**Colnade's advantage:** Named columns with full type information. Works with existing engines (Polars, with more coming). Schema transformations are tracked through `cast_schema`.

### Patito

Patito provides Pydantic-style model classes that serve as Polars DataFrame schemas, with runtime validation. Clean API design, but purely runtime validation — `pl.col("misspelled")` passes the type checker.

**Colnade's advantage:** Static checking of all column references, not just runtime validation. Same schema-as-model pattern, but with compile-time safety.

### Narwhals

Narwhals is a lightweight compatibility layer providing a Polars-like API across multiple backends. It solves "write once, run on any engine" but provides no schema typing or static safety.

**Colnade's advantage:** Full static type safety for column references and schema transformations. Colnade complements the cross-engine approach rather than competing with it.
