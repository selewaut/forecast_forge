import dataclasses

import pytest
import yaml

from forecast_forge.loaders import DataLoadingConfig
from forecast_forge.loaders.dataset_registry import DatasetRegistry, SchemaValidationError


def _write_config(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


def test_get_returns_config(tmp_path):
    _write_config(tmp_path / "walmart.yaml", {
        "loader": {"module": "forecast_forge.loaders.walmart", "class": "WalmartDataLoader"},
        "files": {"train_path": "data/train.csv"},
        "schema": {"unique_id": "group_id", "ds": "date", "y": "weekly_sales"},
    })
    registry = DatasetRegistry(datasets_dir=tmp_path)
    config = registry.get("walmart")
    assert isinstance(config, DataLoadingConfig)
    assert config.loader_class == "WalmartDataLoader"
    assert config.schema["unique_id"] == "group_id"


def test_get_missing_raises_file_not_found(tmp_path):
    registry = DatasetRegistry(datasets_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="walmart"):
        registry.get("walmart")


def test_get_invalid_yaml_raises_value_error(tmp_path):
    (tmp_path / "bad.yaml").write_text("{invalid: yaml: : :}")
    registry = DatasetRegistry(datasets_dir=tmp_path)
    with pytest.raises(ValueError, match="bad"):
        registry.get("bad")


def test_get_unknown_field_raises_type_error(tmp_path):
    _write_config(tmp_path / "bad.yaml", {
        "loader_module": "mod",
        "loader_class": "cls",
        "nonexistent_field": "boom",
    })
    registry = DatasetRegistry(datasets_dir=tmp_path)
    with pytest.raises(TypeError):
        registry.get("bad")


def test_get_default_datasets_dir():
    registry = DatasetRegistry()
    assert registry.datasets_dir.name == "datasets"


def test_validate_passes_valid_dataframe():
    import pandas as pd
    df = pd.DataFrame({
        "unique_id": ["A"], "ds": ["2024-01-01"], "y": [10.0],
    })
    adapter = DatasetRegistry._build_adapter({"unique_id": "id", "ds": "date", "y": "val"})
    # Should not raise
    DatasetRegistry.validate(df, adapter)


def test_validate_rejects_non_numeric_target():
    import pandas as pd
    df = pd.DataFrame({
        "unique_id": ["A"], "ds": ["2024-01-01"], "y": ["abc"],
    })
    adapter = DatasetRegistry._build_adapter({})
    with pytest.raises(SchemaValidationError, match="numeric"):
        DatasetRegistry.validate(df, adapter)


def test_validate_rejects_unparseable_date():
    import pandas as pd
    df = pd.DataFrame({
        "unique_id": ["A"], "ds": ["not-a-date"], "y": [10.0],
    })
    adapter = DatasetRegistry._build_adapter({})
    with pytest.raises(SchemaValidationError, match="date"):
        DatasetRegistry.validate(df, adapter)
