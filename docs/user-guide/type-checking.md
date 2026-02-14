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

This is the most common mistake with string-based DataFrame libraries — a typo in `pl.col("agee")` silently produces wrong results. With Colnade, it's caught immediately.

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

### Nullability mismatch in mapped_from

```python
class Users(Schema):
    age: Column[UInt8 | None]  # nullable

class Bad(Schema):
    age: Column[UInt8] = mapped_from(Users.age)  # non-nullable target
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

**Runtime alternative:** Enable `COLNADE_VALIDATE=1` or `set_validation(True)` and these mismatches are caught at expression construction time:

```python
import colnade
colnade.set_validation(True)

Users.age.fill_null(1.0)
# TypeError: Type mismatch in Users.age.fill_null(): got float value 1.0,
#            expected int for dtype UInt8
```

### Wrong-schema columns in expressions

Expressions erase their source schema. `Orders.amount > 100` and `Users.age > 18` both produce `Expr[Bool]`. The type checker cannot distinguish them:

```python
def process(df: DataFrame[Users]) -> DataFrame[Users]:
    return df.filter(Orders.amount > 100)  # NOT caught — fails at runtime
```

This is a fundamental limitation of the current type system. Column descriptors would need a second type parameter binding them to their schema (requires `TypeVar` defaults from PEP 696).

### Method availability by dtype

All `Column` methods (`.sum()`, `.str_contains()`, `.dt_year()`) are available on every `Column` regardless of dtype. Calling `.sum()` on a string column is not caught statically:

```python
Users.name.sum()  # NOT caught — fails at runtime
```

This requires self-narrowing support in type checkers (e.g., `def sum(self: Column[NumericType]) -> Agg`), which is not yet available.

### Cross-schema equality return type

`Column.__eq__` returns `BinOp[Bool] | JoinCondition` (union type) because it produces a `JoinCondition` for cross-schema comparisons and `BinOp[Bool]` for same-schema comparisons. The `join()` method expects `JoinCondition`, so a `type: ignore[invalid-argument-type]` comment is needed:

```python
users.join(orders, on=Users.id == Orders.user_id)  # type: ignore[invalid-argument-type]
```

## Supported type checkers

| Type Checker | Support Level |
|-------------|---------------|
| [ty](https://github.com/astral-sh/ty) | Full — primary development target |
| mypy | Works — all generic features supported |
| pyright | Works — all generic features supported |
