# Task 0006: Define BaseDataLoader Interface and Schema Adapter

## Status

Planned

## Epic

[Epic 0003: Data Loading Abstraction](../index.md)

## Objective

A `BaseDataLoader` abstract class and `SchemaAdapter` exist in `src/forecast_forge/loaders/` such that any new dataset can be loaded by writing a concrete loader class that implements the interface — no changes to `Forecaster` or the pipeline infrastructure.

## Scope

- Define `BaseDataLoader` abstract class with a `load() -> Dict[str, pd.DataFrame]` method.
- Define `SchemaAdapter` dataclass that maps arbitrary column names to canonical names (`group_id`, `date_col`, `target`, and optional exogenous columns).
- Define `DataLoadingConfig` dataclass: loader class path, file paths, schema mapping, download config.
- Create `src/forecast_forge/loaders/__init__.py` that exports the new types.
- Create `DictLoader` and `CsvLoader` as test-only helpers in `tests/test_loaders/helpers.py`.
- Add a thin adapter in `Forecaster.resolve_source()` that detects a `BaseDataLoader` in `data_conf`, calls `.load()`, and converts the result to a Spark DataFrame before falling back to the hardcoded Walmart path.
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

Expected: 14 tests pass — 5 SchemaAdapter, 6 BaseDataLoader (DictLoader + CsvLoader), 2 DataLoadingConfig, 1 Forecaster integration (Spark). Unit tests: `uv run pytest tests/test_loaders/ -v -m "not spark"`.

## Documentation Updates

- Add `docs/ADRs/0003-dataloader-interface-design.md` (already created).
- Document the loader interface in the README or a new `docs/loaders.md`.

## Test Cases

Ordered by dependency — each group builds on the previous.

### SchemaAdapter tests (`tests/test_loaders/test_schema_adapter.py`)

| # | Test | Input | Expected |
|---|---|---|---|---|
| 1 | Map renames columns to canonical names | DataFrame with columns `store`, `dt`, `sales`; mapping `{"group_id": "store", "date_col": "dt", "target": "sales"}` | Output columns: `group_id`, `date_col`, `target` with same data |
| 2 | Extra columns pass through unchanged | DataFrame with columns `a`, `b`, `c`; mapping only `{"group_id": "a"}` | Output includes `b`, `c` unmodified |
| 3 | Missing required column raises KeyError | DataFrame missing `target` column | `KeyError` with column name in message |
| 4 | Mapping with exogenous columns passes through | DataFrame with columns `id`, `dt`, `y`, `promo`, `holiday`; mapping with `exogenous=["promo", "holiday"]` | `promo`, `holiday` preserved in output |
| 5 | Empty column_map (identity mapping) | DataFrame already has canonical names | Same DataFrame returned, no renames |

### BaseDataLoader tests (`tests/test_loaders/test_base_loader.py`)

Test helpers (in `tests/test_loaders/helpers.py`):
- `DictLoader(data: Dict[str, pd.DataFrame], config: DataLoadingConfig)` — returns the given dict directly, useful for testing the interface contract without I/O.
- `CsvLoader(config: DataLoadingConfig)` — reads CSV files from paths specified in `config.files`, applies the schema adapter, and returns `{"train": DataFrame}`.

| # | Test | Input | Expected |
|---|---|---|---|
| 6 | BaseDataLoader cannot be instantiated directly | `BaseDataLoader(some_config)` | `TypeError` |
| 7 | Concrete loader missing `load()` raises TypeError | Class inheriting BaseDataLoader without implementing `load()` | `TypeError` |
| 8 | DictLoader returns expected dict | Dict of DataFrames | Returns same dict with canonical column names |
| 9 | DictLoader applies SchemaAdapter | DictLoader with mapping `{"group_id": "id"}` | Output has `group_id` column, not `id` |
| 10 | CsvLoader reads CSV files from disk | Temp dir with `train.csv` | Returns `{"train": DataFrame}` with canonical columns |
| 11 | CsvLoader raises FileNotFoundError for missing path | Non-existent CSV path | `FileNotFoundError` |

### DataLoadingConfig tests (`tests/test_loaders/test_loading_config.py`)

| # | Test | Input | Expected |
|---|---|---|---|
| 13 | DataLoadingConfig round-trips through YAML | Config with loader module, class, schema, files | `yaml.safe_load(yaml.dump(config))` recreates same config |
| 14 | SchemaAdapter can be reconstructed from YAML dict | Mapping dict with all fields | SchemaAdapter with correct attributes |

### Forecaster integration tests (`tests/test_loaders/test_forecaster_integration.py`)

Requires Spark (`pytest.mark.spark`).

| # | Test | Input | Expected |
|---|---|---|---|
| 12 | Forecaster accepts a DictLoader via data_conf | `Forecaster(conf, data_conf={"train_data": DictLoader(...)}, spark=spark)` | `resolve_source("train_data")` calls `.load()` and returns Spark DataFrame |

## Notes

The interface design follows MMF's `resolve_source()` which accepts `Union[str, pd.DataFrame, DataFrame]`. The key addition is abstracting the file-to-DataFrame conversion, not the DataFrame-passing itself.

Use `__init_subclass__` or `importlib` registration pattern so loaders can be discovered from the `datasets/` config directory (Task 0008).

Testing strategy: synthetic fixtures — a small CSV on disk, a dict-based loader for unit tests, and a SQLite in-memory loader for SQL-source tests.

## Design Decisions

### Thin adapter in `resolve_source()`

The loader gets into `Forecaster` via `data_conf`:

```python
# Before (current):
Forecaster(conf, data_conf={"train_data": pd.DataFrame(...)})

# After (new):
Forecaster(conf, data_conf={"train_data": WalmartDataLoader(config)})
```

`resolve_source()` detects `BaseDataLoader` instances and calls `.load()`:

```python
def resolve_source(self, key: str):
    if self.data_conf:
        df_val = self.data_conf.get(key)
        if isinstance(df_val, BaseDataLoader):
            result = df_val.load()
            return self.spark.createDataFrame(result["train"])
        elif isinstance(df_val, pd.DataFrame):
            return self.spark.createDataFrame(df_val)
        elif isinstance(df_val, DataFrame):
            return df_val
        else:
            # fallback to hardcoded Walmart (removed in Task 0007)
            return self.spark.createDataFrame(self.load_data())
```

This keeps backward compatibility: old code passing a DataFrame directly still works. New code can pass a loader instance.

All tests in this task should run without Spark (`pytest tests/test_loaders/ -v -m "not spark"`), except the Forecaster integration test (requires `pytest.mark.spark`).
