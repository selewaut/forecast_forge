import pandas as pd
import pytest

from forecast_forge.loaders.base import BaseDataLoader
from forecast_forge.loaders.walmart import WalmartDataLoader, DEFAULT_CONFIG


def test_default_config_has_walmart_paths():
    assert DEFAULT_CONFIG.files is not None
    assert "train_path" in DEFAULT_CONFIG.files
    assert DEFAULT_CONFIG.schema is not None
    assert DEFAULT_CONFIG.schema["unique_id"] == "group_id"


def test_walmart_loader_is_base_dataloader():
    loader = WalmartDataLoader()
    assert isinstance(loader, BaseDataLoader)


def test_walmart_loader_load_returns_merged_dataframe(tmp_path):
    train_csv = tmp_path / "train.csv"
    features_csv = tmp_path / "features.csv"
    stores_csv = tmp_path / "stores.csv"

    pd.DataFrame({
        "Store": [1], "Dept": [1], "Date": ["2010-02-05"],
        "Weekly_Sales": [100.0], "IsHoliday": [False],
    }).to_csv(train_csv, index=False)
    pd.DataFrame({
        "Store": [1], "Date": ["2010-02-05"],
        "Temperature": [50.0], "Fuel_Price": [2.5],
        "CPI": [200.0], "IsHoliday": [False],
    }).to_csv(features_csv, index=False)
    pd.DataFrame({
        "Store": [1], "Type": ["A"], "Size": [100000],
    }).to_csv(stores_csv, index=False)

    import forecast_forge.loaders.walmart as w
    orig_train, orig_features, orig_stores = w.TRAIN_PATH, w.FEATURES_PATH, w.STORES_PATH
    w.TRAIN_PATH = train_csv
    w.FEATURES_PATH = features_csv
    w.STORES_PATH = stores_csv
    try:
        loader = WalmartDataLoader()
        result = loader.load()
    finally:
        w.TRAIN_PATH, w.FEATURES_PATH, w.STORES_PATH = orig_train, orig_features, orig_stores

    assert isinstance(result, pd.DataFrame)
    assert "group_id" in result.columns
    assert "date" in result.columns
    assert "weekly_sales" in result.columns
    assert result["group_id"].iloc[0] == "1_1"
    assert result["weekly_sales"].iloc[0] == 100.0
    assert "temperature" in result.columns
