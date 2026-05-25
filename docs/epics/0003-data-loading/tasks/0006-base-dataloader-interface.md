# Task 0006: Define BaseDataLoader Interface and Schema Adapter

## Status

Planned

## Epic

[Epic 0003: Data Loading Abstraction](../index.md)

## Objective

A `BaseDataLoader` abstract class and `SchemaAdapter` exist in `src/forecast_forge/loaders/` such that any new dataset can be loaded by writing a concrete loader class that implements the interface — no changes to `Forecaster` or the pipeline infrastructure.

## Scope

- Define `BaseDataLoader` abstract class with a `load(train_test_split: bool = True) -> Dict[str, pd.DataFrame]` method.
- Define `SchemaAdapter` dataclass that maps arbitrary column names to canonical names (`group_id`, `date_col`, `target`, and optional exogenous columns).
- Define `DataLoadingConfig` dataclass: loader class path, file paths or SQL connection, schema mapping, download logic flag.
- Create `src/forecast_forge/loaders/__init__.py` that exports the new types.
- Keep `Forecaster.resolve_source()` working with the new interface via a thin adapter.
- Existing hardcoded `config.py` and `data_loader.py` remain untouched until Task 0007.

## Acceptance Criteria

- `BaseDataLoader` is an abstract class using `abc.ABC` with a concrete `load()` signature.
- `SchemaAdapter` accepts a dict mapping like `{"group_id": "store_dept", "date_col": "date", "target": "sales"}` and exposes `.map(df) -> DataFrame` that renames columns and validates presence.
- `DataLoadingConfig` can be serialized from/deserialized to YAML.
- A unit test validates that a minimal toy loader (e.g., `DictLoader` returning hardcoded data) works with `Forecaster.resolve_source()`.
- No changes to the Walmart pipeline yet — backward compatibility is maintained.

## Validation

```sh
uv run pytest tests/test_loaders/ -v
```

Expected: test_base_loader_interface, test_schema_adapter_mapping, test_schema_adapter_missing_column, test_loading_config_from_yaml all pass.

## Documentation Updates

- Add `docs/ADRs/0003-dataloader-interface-design.md` (already created).
- Document the loader interface in the README or a new `docs/loaders.md`.

## Notes

The interface design follows MMF's `resolve_source()` which accepts `Union[str, pd.DataFrame, DataFrame]`. The key addition is abstracting the file-to-DataFrame conversion, not the DataFrame-passing itself.

Use `__init_subclass__` or `importlib` registration pattern so loaders can be discovered from the `datasets/` config directory (Task 0008).

Testing strategy: synthetic fixtures — a small CSV on disk, a dict-based loader for unit tests, and a SQLite in-memory loader for SQL-source tests.
