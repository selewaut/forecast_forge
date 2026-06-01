import pandas as pd
import pytest

from forecast_forge.datasets.walmart import load_walmart_data


def test_load_walmart_data_returns_dataframe(tmp_path):
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
        "MarkDown1": [None],
    }).to_csv(features_csv, index=False)
    pd.DataFrame({
        "Store": [1], "Type": ["A"], "Size": [100000],
    }).to_csv(stores_csv, index=False)

    import forecast_forge.datasets.walmart as w
    orig_train, orig_features, orig_stores = w.TRAIN_PATH, w.FEATURES_PATH, w.STORES_PATH
    try:
        w.TRAIN_PATH = train_csv
        w.FEATURES_PATH = features_csv
        w.STORES_PATH = stores_csv

        result = load_walmart_data()
    finally:
        w.TRAIN_PATH, w.FEATURES_PATH, w.STORES_PATH = orig_train, orig_features, orig_stores

    assert isinstance(result, pd.DataFrame)
    assert "group_id" in result.columns
    assert "date" in result.columns
    assert "weekly_sales" in result.columns
    assert "week" in result.columns
    assert "temp_bin" in result.columns
    assert result["group_id"].iloc[0] == "1_1"
    assert result["weekly_sales"].iloc[0] == 100.0


def test_negative_sales_clipped_to_zero(tmp_path):
    train_csv = tmp_path / "train.csv"
    features_csv = tmp_path / "features.csv"
    stores_csv = tmp_path / "stores.csv"

    pd.DataFrame({
        "Store": [1], "Dept": [1], "Date": ["2010-02-05"],
        "Weekly_Sales": [-50.0], "IsHoliday": [False],
    }).to_csv(train_csv, index=False)
    pd.DataFrame({
        "Store": [1], "Date": ["2010-02-05"],
        "Temperature": [50.0], "Fuel_Price": [2.5],
        "CPI": [200.0], "IsHoliday": [False],
        "MarkDown1": [None],
    }).to_csv(features_csv, index=False)
    pd.DataFrame({
        "Store": [1], "Type": ["A"], "Size": [100000],
    }).to_csv(stores_csv, index=False)

    import forecast_forge.datasets.walmart as w
    orig_train, orig_features, orig_stores = w.TRAIN_PATH, w.FEATURES_PATH, w.STORES_PATH
    try:
        w.TRAIN_PATH = train_csv
        w.FEATURES_PATH = features_csv
        w.STORES_PATH = stores_csv

        result = load_walmart_data()
    finally:
        w.TRAIN_PATH, w.FEATURES_PATH, w.STORES_PATH = orig_train, orig_features, orig_stores

    assert result["weekly_sales"].iloc[0] == 0.0
