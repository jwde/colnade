# Expressions

Colnade builds expression trees from column operations. These trees are translated by backend adapters into engine-native code.

## Comparisons

All comparisons return `Expr[Bool]`:

```python
Users.age > 18       # greater than
Users.age < 65       # less than
Users.age >= 18      # greater or equal
Users.age <= 65      # less or equal
Users.age == 30      # equal
Users.age != 0       # not equal
```

## Arithmetic

Arithmetic preserves the column's dtype:

```python
Users.score * 2          # multiply
Users.score / 100        # divide
Users.age + 1            # add
Users.age - 1            # subtract
Users.age % 10           # modulo
-Users.score             # negate
```

Reverse operators work too: `1 + Users.age` produces the same tree as `Users.age + 1`.

## Logical operators

Combine boolean expressions with `&` (and) and `|` (or):

```python
(Users.age > 25) & (Users.score > 80)     # and
Users.name.str_starts_with("A") | Users.name.str_starts_with("E")  # or
```

!!! warning "Use parentheses"
    Python's operator precedence requires parentheses around each comparison when using `&` and `|`.

## Aggregations

Aggregation methods produce `Agg` nodes for use in `group_by().agg()`:

```python
Users.score.sum()       # sum of values
Users.score.mean()      # mean (returns Float64)
Users.score.min()       # minimum
Users.score.max()       # maximum
Users.id.count()        # count non-null (returns UInt32)
Users.score.std()       # standard deviation
Users.score.var()       # variance
Users.score.first()     # first value
Users.score.last()      # last value
Users.name.n_unique()   # unique count (returns UInt32)
```

Use `.alias()` to bind aggregation results to output columns:

```python
class Stats(Schema):
    name: Column[Utf8]
    avg_score: Column[Float64]
    user_count: Column[UInt32]

df.group_by(Users.name).agg(
    Users.score.mean().alias(Stats.avg_score),
    Users.id.count().alias(Stats.user_count),
)
```

## String methods

Available on `Column[Utf8]`:

```python
Users.name.str_contains("ali")        # substring search
Users.name.str_starts_with("A")       # prefix check
Users.name.str_ends_with("ce")        # suffix check
Users.name.str_len()                  # string length
Users.name.str_to_lowercase()         # to lowercase
Users.name.str_to_uppercase()         # to uppercase
Users.name.str_strip()                # strip whitespace
Users.name.str_replace("old", "new")  # replace substring
```

## Temporal methods

Available on `Column[Datetime]`:

```python
Events.timestamp.dt_year()            # extract year
Events.timestamp.dt_month()           # extract month
Events.timestamp.dt_day()             # extract day
Events.timestamp.dt_hour()            # extract hour
Events.timestamp.dt_minute()          # extract minute
Events.timestamp.dt_second()          # extract second
Events.timestamp.dt_truncate("1d")    # truncate to interval
```

## Null handling

```python
Users.age.is_null()                   # check if null
Users.age.is_not_null()               # check if not null
Users.age.fill_null(0)                # replace nulls with value
Users.age.assert_non_null()           # assert non-null (runtime)
```

## NaN handling

Available on float columns:

```python
Users.score.is_nan()                  # check if NaN
Users.score.fill_nan(0.0)             # replace NaN with value
```

## Sort expressions

Control sort direction:

```python
df.sort(Users.score.desc())           # descending
df.sort(Users.name.asc())             # ascending (explicit)
df.sort(Users.name)                   # ascending (default)
```

## Aliasing

Bind an expression result to a target column:

```python
(Users.score * 2).alias(Users.score)           # overwrite score
Users.score.mean().alias(Users.score)            # alias aggregation
```

## Casting

Cast a column to a different type:

```python
Users.id.cast(Float64)                # cast UInt64 → Float64
```

## Window functions

Apply an aggregation within each partition and broadcast the result back to every row:

```python
Users.score.sum().over(Users.name)    # per-name total, broadcast to each row
Users.score.mean().over(Users.name)   # per-name average
```
