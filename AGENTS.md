# Colnade — Implementation Plan

This document tracks the Phase 1 implementation of Colnade: a statically type-safe DataFrame abstraction layer over existing engines. Phase 1 delivers the core library (`colnade`) and the Polars reference adapter (`colnade-polars`).

Full specification: [`spec.md`](spec.md)

---

## Testing Philosophy

**Every PR is self-contained: implement + test + type-check.** There is no deferred testing.

Each issue (#2–#9) carries its own rigorous test requirements:

- **Unit tests** for all new runtime behavior, with specific test cases enumerated in the issue
- **Static type tests** (`tests/typing/`) for all new public API surface — verified by ty
- **CI must be green** before any PR merges

Issues #10 and #11 cover **cross-cutting concerns only** — multi-layer type checking scenarios, end-to-end pipelines, and coverage analysis. They do not carry the burden of basic module-level testing; that's already done.

---

## Task Sequence

Tasks are ordered by dependency. Each task is implemented as an independent branch/PR, merged only after implementation, testing, QA, and documentation are complete.

### Dependency Graph

```
#1 Scaffolding
 └─► #2 Data Types
      └─► #3 Schema Layer
           └─► #4 Expression DSL
                ├─► #5 Nested Types (Struct/List)
                │    │
                │    ▼
                └─► #6 DataFrame & LazyFrame ◄── #5
                     └─► #7 Join System
                          └─► #8 cast_schema & Validation
                               └─► #9 Backend Protocol & Polars Adapter
                                    ├─► #10 Cross-cutting Type Tests
                                    └─► #11 Cross-cutting Integration Tests
                                         │
                                    #10 ──┤
                                          ▼
                                         #12 Documentation
```

---

## Tasks

### #1 — [Project scaffolding & build system](https://github.com/jwde/colnade/issues/1)

| | |
|---|---|
| **Scope** | Monorepo structure (`colnade/` + `colnade-polars/`), `pyproject.toml` for both packages, CI pipeline (GitHub Actions), dev tooling (ruff, pytest), `py.typed` marker |
| **Tests** | CI infrastructure: smoke test for imports, ty check baseline on empty typing test, ruff passing |
| **Blocked by** | — |
| **Spec sections** | §3.2 Package Structure, §3.3 Dependency Policy |

### #2 — [Data types & internal typing utilities](https://github.com/jwde/colnade/issues/2)

| | |
|---|---|
| **Scope** | `dtypes.py`: all dtype sentinels (Bool, UInt8, ..., Float64, Utf8, Datetime, Struct, List). `_types.py`: type category unions (NumericType, TemporalType), core TypeVars |
| **Tests** | Unit: class identity, generic parameterization, type categories. Static: valid dtype usage accepted, generics accepted. ty check passes. |
| **Blocked by** | #1 |
| **Spec sections** | §3.2, §4.8, §5.3, §15.1 |

### #3 — [Schema layer — metaclass, Column descriptors, inheritance](https://github.com/jwde/colnade/issues/3)

| | |
|---|---|
| **Scope** | `schema.py`: `SchemaMeta` metaclass, `Schema` base class (extends Protocol), `Column[DType, SchemaType]` descriptor, schema inheritance/composition, nullable column annotations |
| **Tests** | Unit: schema creation, Column properties, inheritance, trait composition, nullable columns, edge cases. Static: `Users.age` accepted, `Users.agee` rejected, structural subtyping with bounded TypeVar. ty check passes. |
| **Blocked by** | #2 |
| **Spec sections** | §4.1–4.5, §4.7.1 |

### #4 — [Expression DSL — AST nodes, Column expression building, null propagation](https://github.com/jwde/colnade/issues/4)

| | |
|---|---|
| **Scope** | `expr.py`: all AST nodes (Expr, ColumnRef, BinOp, UnaryOp, Literal, FunctionCall, Agg, AliasedExpr, SortExpr). Column operator overloads (comparison, arithmetic, logical). Aggregation methods. String/temporal/null/NaN methods with type-conditional availability via `self` narrowing. Null propagation through expression types. `lit()` helper |
| **Tests** | Unit: every AST node construction, every operator overload, every Column method (agg, string, temporal, null, NaN, general), expression chaining. Static: method availability by dtype (sum on numeric OK, sum on Utf8 rejected), null propagation types, expression type correctness, AliasedExpr type matching. ty check passes. |
| **Blocked by** | #3 |
| **Spec sections** | §5.1–5.3, §4.3, §4.7.2–4.7.4, §15.4 |

### #5 — [Nested types — Struct[S] and List[T]](https://github.com/jwde/colnade/issues/5)

| | |
|---|---|
| **Scope** | `Struct[S]` and `List[T]` as column types. `.field()` method (only on Struct columns, schema-constrained). `ListAccessor` class with `.list` property (only on List columns). `StructFieldAccess` and `ListOp` AST nodes. Nested nullability propagation |
| **Tests** | Unit: Struct/List in schemas, .field() AST, .list.* AST, chained operations, nested nullability. Static: .field() only on Struct, .list only on List, schema constraints on .field(), numeric-only ListAccessor methods, nullable struct propagation. ty check passes. |
| **Blocked by** | #4 |
| **Spec sections** | §4.8.1–4.8.4, §5.1, §8.3 |

### #6 — [DataFrame & LazyFrame interfaces](https://github.com/jwde/colnade/issues/6)

| | |
|---|---|
| **Scope** | `dataframe.py`: `DataFrame[S]`, `LazyFrame[S]` with all schema-preserving ops (filter, sort, limit, head, tail, sample, unique, drop_nulls, with_columns). Select overloads (arities 1–10). `GroupBy[S]` / `LazyGroupBy[S]`. `.lazy()` / `.collect()` conversions. `UntypedDataFrame` / `UntypedLazyFrame` escape hatch |
| **Tests** | Unit: construction, return types for all ops, GroupBy, conversions, LazyFrame restrictions (no head/tail/sample). Static: schema preservation, select rejects wrong-schema columns, sort/group_by column validation, lazy vs eager distinction (LazyFrame not assignable to DataFrame), untyped escape. ty check passes. |
| **Blocked by** | #4, #5 |
| **Spec sections** | §6.1–6.3, §6.6, §6.7, §12.3, §15.3 |

### #7 — [Join system — JoinCondition, JoinedDataFrame, JoinedLazyFrame](https://github.com/jwde/colnade/issues/7)

| | |
|---|---|
| **Scope** | `JoinCondition` class. `Column.__eq__` overload (same schema → `Expr[Bool]`, different schema → `JoinCondition`). `JoinedDataFrame[S, S2]` and `JoinedLazyFrame[S, S2]` with select overloads accepting `Column[Any, S] \| Column[Any, S2]`. `.join()` method on DataFrame/LazyFrame |
| **Tests** | Unit: JoinCondition creation, == dispatch (same vs different schema), JoinedDataFrame ops, conversions, join how variants. Static: joined select accepts both schemas, rejects third schema, JoinedDataFrame not assignable to DataFrame, join return types. ty check passes. |
| **Blocked by** | #6 |
| **Spec sections** | §6.1, §6.5 |

### #8 — [cast_schema, mapped_from & runtime schema validation](https://github.com/jwde/colnade/issues/8)

| | |
|---|---|
| **Scope** | `cast_schema()` on all four frame types. `mapped_from()` field modifier. Name resolution (mapping → mapped_from → name match). `SchemaError` exception. Runtime validation (columns, types, nullability). `extra="drop"/"forbid"`. Nullability enforcement at cast boundaries |
| **Tests** | Unit: mapped_from storage, cast_schema name matching, mapped_from resolution, explicit mapping, resolution precedence, SchemaError fields (missing/extra/type mismatch/null violations), nullability enforcement, cast on JoinedDataFrame. Static: return types, mapped_from type mismatch rejected, nullability narrowing rejected. ty check passes. |
| **Blocked by** | #7 |
| **Spec sections** | §4.6, §4.7.5, §6.4, §6.5, §12.1–12.2 |

### #9 — [Backend protocol & Polars adapter](https://github.com/jwde/colnade/issues/9)

| | |
|---|---|
| **Scope** | `_protocols.py`: `BackendProtocol` interface. `colnade_polars/adapter.py`: `PolarsAdapter` with full expression translation (all AST nodes → Polars exprs), all execution methods, dtype mapping. `colnade_polars/io.py`: read/write/scan parquet/CSV. Wire DataFrame/LazyFrame to delegate to adapter |
| **Tests** | Unit: expression translation for every AST node type, dtype mapping. Integration: execution through Polars with real data (filter, select, sort, limit, unique, drop_nulls, with_columns, group_by+agg, join), I/O (read/write parquet/CSV, schema mismatch errors), nested type operations through Polars, end-to-end pipeline test. |
| **Blocked by** | #8 |
| **Spec sections** | §8.1–8.4, §12.1–12.2, §15.5 |

### #10 — [Cross-cutting static type checking tests](https://github.com/jwde/colnade/issues/10)

| | |
|---|---|
| **Scope** | Multi-layer type checking scenarios: full pipeline type flow, generic function patterns (§7), systematic §10 coverage matrix verification, error message documentation |
| **Tests** | Static: pipeline types across layers, passthrough/constrained/column-parameterized generics, all ✅ rows in §10 matrix, error message capture. ty check passes on all `tests/typing/` files. |
| **Blocked by** | #9 |
| **Spec sections** | §7, §10, §11.1–11.6, §14.2 |

### #11 — [Cross-cutting integration tests & end-to-end pipelines](https://github.com/jwde/colnade/issues/11)

| | |
|---|---|
| **Scope** | End-to-end pipeline tests: Users pipeline, aggregation pipeline, multi-join pipeline, nested types pipeline, null handling pipeline, lazy pipeline. Cross-layer integration. Edge cases. Coverage analysis (>90% target). |
| **Tests** | E2E: 6 pipeline tests with result verification. Integration: cross-layer roundtrips, edge cases (column collision, empty DataFrame, long expression chains). Coverage: >90% line coverage on both packages. |
| **Blocked by** | #9 |
| **Spec sections** | §14.1, §14.3 |

### #12 — [Documentation & examples](https://github.com/jwde/colnade/issues/12)

| | |
|---|---|
| **Scope** | README.md (overview, install, quick start, features, comparison). API reference. Example files (basic usage, null handling, joins, generics, nested types, full pipeline). Type checker error showcase |
| **Tests** | All example files are executable and run in CI. All examples pass type checking (ty check). |
| **Blocked by** | #10, #11 |
| **Spec sections** | §1, §2, §7, §11, §13 Phase 1 deliverables |

---

## Workflow

Each task follows this process:

1. **Branch** from `main`: `git checkout -b issue-N-short-description`
2. **Implement** the scope described in the issue
3. **Test** — unit tests pass, static type tests pass (ty check), CI green
4. **QA** — review against spec references, verify every acceptance criterion
5. **Document** — update the issue with implementation notes, decisions, and test results
6. **PR** — open pull request, reference the issue (`Closes #N`)
7. **Merge** — squash-merge to `main` after review

**A PR cannot merge unless CI is green.** This means all unit tests, static type tests, and linting pass. No exceptions.

Tasks #10 and #11 can proceed in parallel after #9 is complete.
