# Type Checker Integration

Colnade is designed to work with standard Python type checkers. This page shows what errors are caught and the actual error messages you'll see.

## Errors caught statically

### Misspelled column name

```python
x = Users.agee
```

```
error[unresolved-attribute]: Class `Users` has no attribute `agee`
  --> example.py:15:5
   |
15 | x = Users.agee
   |     ^^^^^^^^^^
```

With string-based DataFrame libraries, a typo in `pl.col("agee")` isn't caught until your code runs. With Colnade, it's caught in your editor before you run anything.

### Schema mismatch at function boundary

```python
df: DataFrame[Users] = read_parquet("users.parquet", Users)
wrong: DataFrame[Orders] = df
```

```
error[invalid-assignment]: Object of type `DataFrame[Users]` is not assignable
to `DataFrame[Orders]`
  --> example.py:19:8
   |
19 | wrong: DataFrame[Orders] = df
   |        -----------------   ^^ Incompatible value of type `DataFrame[Users]`
```

Generic invariance ensures that schemas are not accidentally swapped between functions.

### Method availability by dtype

Column methods are restricted to appropriate types. Calling `.sum()` on a string column is caught at type-check time:

```python
Users.name.sum()
```

```
error[invalid-argument-type]: Argument to bound method `sum` is incorrect
  --> example.py:10:5
   |
10 | _ = Users.name.sum()
   |     ^^^^^^^^^^^^^^^^ Expected numeric Column, found `Column[Utf8]`
```

This applies to numeric aggregations (`sum`, `mean`, `std`, `var`), NaN methods (`is_nan`, `fill_nan` — float only), string methods (`str_contains`, `str_len`, etc. — Utf8 only), temporal methods (`dt_year`, `dt_hour`, etc.), and struct field access (`.field()` — Struct only).

### Nullability mismatch in mapped_from

```python
class Users(cn.Schema):
    age: cn.Column[cn.UInt8 | None]  # nullable

class Bad(cn.Schema):
    age: cn.Column[cn.UInt8] = cn.mapped_from(Users.age)  # non-nullable target
```

```
error[invalid-assignment]: Object of type `Column[UInt8 | None]` is not
assignable to `Column[UInt8]`
  --> example.py:23:10
   |
23 |     age: Column[UInt8] = mapped_from(Users.age)
   |          -------------   ^^^^^^^^^^^^^^^^^^^^^^
```

This prevents silently mapping nullable data into a non-nullable schema.

## Known limitations

### Literal value types in expressions

Python's type system cannot enforce that literal values match column dtypes. All operator and method parameters accept `Any`:

```python
Users.age.fill_null(1.0)  # NOT caught — float for UInt8 column
Users.age + "hello"       # NOT caught — string added to integer
Users.name > 42           # NOT caught — int compared to string
```

This is a fundamental limitation — Python lacks type-level functions to map `Column[UInt8]` → `fill_null(value: int)`. It would require associated types or conditional types, which Python does not support.

**Runtime alternative:** Enable validation and these mismatches are caught at expression construction time:

```python
import colnade as cn

cn.set_validation(cn.ValidationLevel.STRUCTURAL)
# or: COLNADE_VALIDATE=structural

Users.age.fill_null(1.0)
# TypeError: Type mismatch in Users.age.fill_null(): got float value 1.0,
#            expected int for dtype UInt8
```

Use `ValidationLevel.FULL` (or `COLNADE_VALIDATE=full`) to also enforce `Field()` value constraints at data boundaries. See [DataFrames: Validation](dataframes.md#validation) for details.

### Wrong-schema columns in expressions

Expressions erase their source schema. `Orders.amount > 100` and `Users.age > 18` both produce `Expr[Bool]`. The type checker cannot distinguish them:

```python
def process(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.filter(Orders.amount > 100)  # NOT caught statically
```

This is a fundamental limitation of the current type system. Column descriptors would need a second type parameter binding them to their schema (requires `TypeVar` defaults from PEP 696).

**Runtime alternative:** When validation is enabled (`STRUCTURAL` or `FULL`), DataFrame and LazyFrame operations validate that all column references in an expression belong to the frame's schema. The example above raises `SchemaError` at runtime:

```python
cn.set_validation(cn.ValidationLevel.STRUCTURAL)

df.filter(Orders.amount > 100)
# cn.SchemaError: Missing columns: amount
```

This guard covers `filter`, `sort`, `with_columns`, `select`, `unique`, `drop_nulls`, `group_by`, and `agg`. On `JoinedDataFrame`/`JoinedLazyFrame`, columns from either schema are accepted.

### List operations return untyped results

The `.list` property on a `Column[List[DType]]` returns a `ListAccessor` whose methods (`.len()`, `.get()`, `.sum()`, etc.) return `ListOp[Any]` rather than a precisely typed result. This means the type checker won't catch misuse like calling `.sum()` on a list of strings:

```python
UserProfile.tags.list.sum()   # NOT caught — tags is List[Utf8], sum makes no sense
```

This is a current limitation of Python type checkers — property-based self-narrowing (needed to restrict `.list` to `Column[List[...]]`) is not yet supported. The annotations are in place and will become precise when type checker support improves.

### Column-to-column equality is treated as a join key

`Column.__eq__` is overloaded so that comparing two columns (`Users.id == Orders.user_id`) is typed as a `JoinCondition`, while comparing a column to a value (`Users.age == 30`) is typed as `BinOp[Bool]`. This means both the common cases type-check without suppressions:

```python
users.join(orders, on=Users.id == Orders.user_id)   # JoinCondition — accepted by join()
df.filter(Users.age == 30)                            # BinOp[Bool] — accepted by filter()
df.filter((Users.age == 30) & (Users.name == "Alice"))
```

The one edge case: a *same-schema* column-to-column comparison (`Users.age == Users.score`) returns `BinOp[Bool]` at runtime but is statically typed as `JoinCondition`, so passing it to `filter()` is rejected by the type checker. This is rare; rewrite it as an explicit comparison or add a `type: ignore` at that call site if needed.

## Supported type checkers

| Type Checker | Support Level |
|-------------|---------------|
| [ty](https://github.com/astral-sh/ty) | Full — primary development target |
| mypy | Works — all generic features supported |
| pyright | Works — all generic features supported |
