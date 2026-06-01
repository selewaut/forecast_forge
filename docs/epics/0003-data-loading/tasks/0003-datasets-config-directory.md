# Task 0003: Add Datasets Config Directory and Schema Validation

## Status

**Superseded** — the `datasets/` YAML config directory and `DatasetRegistry` were removed. Dataset configuration is now handled via per-dataset Python modules (e.g., `datasets/walmart/`). Schema validation (`SchemaValidationError`) was removed along with `DatasetRegistry`.

## Epic

[Epic 0003: Data Loading Abstraction](../index.md)

## Objective

A `datasets/` directory at the project root contains YAML files declaring available datasets. Each file specifies loader class, file paths or SQL connection, schema mapping, and download instructions. The pipeline selects a dataset by name at runtime — no hardcoded imports.

## Scope

- Create `datasets/` directory at project root.
- Create `datasets/walmart.yaml` with Walmart dataset configuration: loader, paths, schema mapping, Kaggle competition ID.
- Add a `DatasetRegistry` that scans `datasets/` and returns `DataLoadingConfig` by dataset name.
- Add schema validation on load: required columns exist, date column is parseable, target column is numeric, group_id column has no nulls.
- Integration: `Forecaster.__init__` accepts a `dataset_name` string that replaces the hardcoded `load_data()` call.

## Acceptance Criteria

- `datasets/walmart.yaml` exists and fully describes the Walmart dataset.
- `DatasetRegistry.get("walmart")` returns a valid `DataLoadingConfig`.
- `Forecaster("config.yaml", dataset_name="walmart")` loads and preprocesses data without referencing `data_loader.py` directly.
- Schema validation rejects a config with missing required columns and provides a clear error message.
- A second dummy dataset config can be added and loaded by changing only the `dataset_name` parameter.

## Validation

```sh
uv run python -c "
from forecast_forge.loaders import DatasetRegistry
cfg = DatasetRegistry.get('walmart')
print(cfg)
"
```

Expected: prints the resolved config with loader class, paths, and schema mapping.

```sh
uv run python -c "
from forecast_forge import Forecaster
from omegaconf import OmegaConf
f = Forecaster(OmegaConf.create({'dataset_name': 'walmart', ...}))
"
```

Expected: loads and preprocesses Walmart data without errors.

## Documentation Updates

- Add `datasets/README.md` explaining how to add a new dataset.
- README updated with dataset configuration section.

## Gotchas

### New methods must not split `__init__`

When adding a method to `Forecaster`, placing a `def` between field assignments and the experiment_id/run_date blocks causes Python to **end `__init__` early**. Any lines after the method definition are interpreted as class-level dead code:

```python
# BAD — logic after def ends up outside __init__
class Forecaster:
    def __init__(self, ...):
        self.spark = spark
        self._loader = self._resolve_loader(dataset_name)  # ← triggers def

    def _resolve_loader(self, ...):  # ← this def ENDS __init__
        ...

        if experiment_id:              # ← DEAD CODE — never runs
            self.experiment_id = ...
```

Fix: keep the experiment_id + run_date logic **before** any `def` statements inside `__init__`.

## Notes

The `DatasetRegistry` should use a simple directory scan + `yaml.safe_load` — no need for a database or complex indexing. A dataset name maps 1:1 to `datasets/{name}.yaml`.

YAML structure for `walmart.yaml`:

```yaml
loader:
  module: forecast_forge.loaders.walmart
  class: WalmartDataLoader
files:
  train_path: data/walmart_sales_forecasting/train.csv
  test_path: data/walmart_sales_forecasting/test.csv
  features_path: data/walmart_sales_forecasting/features.csv
  stores_path: data/walmart_sales_forecasting/stores.csv
schema:
  group_id: store_dept    # constructed from store + dept
  date_col: date
  target: weekly_sales
  exogenous:
    - isholiday
    - temperature
    - fuel_price
    - cpi
download:
  source: kaggle
  competition: walmart-recruiting-store-sales-forecasting
```
