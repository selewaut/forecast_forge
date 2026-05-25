# ADR 0003: DataLoader Interface Design

## Status

Proposed

## Context

The current pipeline hardcodes Walmart dataset loading in `data_loader.py`, `data.py`, and `config.py`. Making the pipeline dataset-agnostic requires an abstraction for loading data from arbitrary sources (CSV, Parquet, SQL, Delta) with configurable schema mapping.

The existing `Forecaster.resolve_source()` already accepts `Union[str, pd.DataFrame, DataFrame]` for in-memory data, but the file-to-DataFrame conversion path is hardcoded.

## Decision

Use an abstract base class pattern for loaders:

```python
class BaseDataLoader(ABC):
    @abstractmethod
    def load(self, split: bool = True) -> Dict[str, pd.DataFrame]:
        ...
```

Concrete loaders (`CsvLoader`, `ParquetLoader`, `WalmartDataLoader`) implement the interface. A `SchemaAdapter` handles column renaming/validation after load. A `DatasetRegistry` scans a `datasets/` directory and resolves dataset names to `DataLoadingConfig` objects.

Key decisions:

1. **Loader discovery by convention, not registration** — `datasets/{name}.yaml` maps 1:1 to dataset names. No central registry to update.
2. **Schema mapping at load time, not query time** — The loader returns DataFrames with canonical column names. The downstream pipeline never sees original column names.
3. **Download is optional** — `BaseDataLoader` has an optional `download()` method. Only loaders that need remote data (like Walmart/Kaggle) implement it.
4. **No SQL abstraction yet** — SQL source support is deferred until there's a concrete use case. The interface is designed to allow it but not enforced.

## Consequences

- Easier: adding a new dataset means writing a loader class + YAML config, no pipeline changes.
- Easier: testing with synthetic data — use a `DictLoader` or `CsvLoader` with a temp file.
- Harder: legacy `data_loader.py` must be migrated in lockstep — the old code and new interface coexist briefly during Task 0007.
- Neutral: the `Forecaster.load_data()` method grows a lookup step (`DatasetRegistry.get(dataset_name)`) but the rest of the pipeline stays the same.
