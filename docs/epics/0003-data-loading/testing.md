# Epic 0003: Testing Strategy

## Approach

Use small synthetic fixtures — never depend on the real Walmart dataset for unit tests. Integration tests use a minimal CSV with 2 groups, 52 weeks of data each.

## Test Layers

### Unit Tests (pytest, no Spark)

| Test File | What It Covers | Fixture |
|---|---|---|
| `tests/test_loaders/test_base_loader.py` | `BaseDataLoader` ABC enforces `load()` contract; `DictLoader` toy implementation works | In-memory dict |
| `tests/test_loaders/test_schema_adapter.py` | Column renaming, missing column raises `KeyError`, extra columns pass through, date dtype coercion | Small DataFrame (5 rows) |
| `tests/test_loaders/test_dataset_registry.py` | Registry scans `datasets/` dir, loads YAML, returns `DataLoadingConfig`, missing file raises | Temporary YAML files |
| `tests/test_loaders/test_config_validation.py` | Schema validation rejects missing columns, bad dtypes, null group_ids | DataFrame with intentional errors |

### Integration Tests (pytest, no Spark)

| Test File | What It Covers | Fixture |
|---|---|---|
| `tests/test_loaders/test_walmart_loader.py` | `WalmartDataLoader` produces same columns as old `load_data()` (using a small cached CSV, not live Kaggle) | Pre-downloaded 100-row sample of Walmart data |
| `tests/test_end_to_end.py` | `Forecaster` with `dataset_name="test_dataset"` loads data, runs backtest, returns metrics | Synthetic 2-group CSV + `datasets/test_dataset.yaml` |

### Smoke Tests (Spark required)

| Script | What It Covers |
|---|---|
| `make run` after refactor | Full pipeline with Walmart data produces identical evaluation output |

## Fixture Strategy

- `conftest.py` at `tests/` level provides:
  - `synthetic_weekly_data()` — 2 groups, 52 weeks, known seasonal pattern
  - `synthetic_dataset_config()` — temp dir with CSV + YAML config
  - `walmart_sample()` — small pickled subset of real Walmart data (committed to repo under `tests/fixtures/`)

## Running Tests

```sh
# Unit only (fast, no Spark)
uv run pytest tests/ -v -m "not spark"

# All tests
uv run pytest tests/ -v
```

## CI Recommendation (future)

Add a GitHub Actions workflow that:
1. Runs unit tests on every push
2. Runs integration + smoke tests on PRs targeting `main`
