# Epic 0003: Testing Strategy

## Approach

Use small synthetic fixtures — never depend on the real Walmart dataset for unit tests. Integration tests use a minimal CSV with 2 groups, 52 weeks of data each.

## Test Layers

### Unit Tests (pytest, no Spark)

| Test File | What It Covers | Fixture |
|---|---|---|
| `tests/test_datasets/test_walmart_data.py` | `load_walmart_data()` returns merged DataFrame with expected columns; negative sales clipped to zero | Temporary CSV files via `tmp_path` |

### Smoke Tests (Spark required)

| Script | What It Covers |
|---|---|
| `make run` | Full pipeline with Walmart data produces evaluation output |

## Fixture Strategy

- `conftest.py` at `tests/` level provides:
  - `synthetic_weekly_data()` — 2 groups, 52 weeks, known seasonal pattern
  - `canonical_weekly_data()` — same data with canonical column names

## Running Tests

```sh
# All tests
uv run pytest tests/ -v
```

## CI Recommendation (future)

Add a GitHub Actions workflow that:
1. Runs unit tests on every push
2. Runs integration + smoke tests on PRs targeting `main`
