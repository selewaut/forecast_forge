import os
import zipfile
from pathlib import Path

import kaggle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "walmart_sales_forecasting"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
STORES_PATH = DATA_DIR / "stores.csv"

MARKDOWN_FEATURES = ["markdown1", "markdown2", "markdown3", "markdown4", "markdown5"]
TEMPERATURE_BINS = [-np.inf, 40, 55, 70, 85, 95, np.inf]
TEMPERATURE_LABELS = [0, 1, 2, 3, 4, 5]


def _download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        competition="walmart-recruiting-store-sales-forecasting",
        path=DATA_DIR,
        force=True,
    )
    zip_path = DATA_DIR / "walmart-recruiting-store-sales-forecasting.zip"
    if zip_path.exists():
        _unzip(zip_path, DATA_DIR)
        zip_path.unlink()


def _unzip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        for file in zip_ref.namelist():
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(extract_to, file)
                _unzip(nested_zip_path, extract_to)
                os.remove(nested_zip_path)


def _clip_negative_sales(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[target_col] = df[target_col].clip(0)
    return df


def _add_temperature_bins(df: pd.DataFrame, temp_col: str = "temperature") -> pd.DataFrame:
    df["temp_bin"] = pd.cut(df[temp_col], bins=TEMPERATURE_BINS, labels=TEMPERATURE_LABELS)
    df = df.drop(columns=[temp_col])
    return df


def _generate_week_feature(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df["week"] = df[date_col].dt.isocalendar().week
    return df


def _get_markdown_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in MARKDOWN_FEATURES if col in df.columns]


def load_walmart_data() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        _download()

    df_train = pd.read_csv(TRAIN_PATH)
    df_features = pd.read_csv(FEATURES_PATH)
    df_stores = pd.read_csv(STORES_PATH)

    df_train.columns = df_train.columns.str.lower()
    df_features.columns = df_features.columns.str.lower()
    df_stores.columns = df_stores.columns.str.lower()

    df_train["date"] = pd.to_datetime(df_train["date"])
    df_features["date"] = pd.to_datetime(df_features["date"])

    df_train["group_id"] = (
        df_train["store"].astype(str) + "_" + df_train["dept"].astype(str)
    )

    df_train = df_train.merge(
        df_features, on=["date", "store", "isholiday"], how="left"
    )
    df_train = df_train.merge(df_stores, on=["store"], how="left")

    df_train = _generate_week_feature(df_train, "date")
    df_train = _clip_negative_sales(df_train, "weekly_sales")
    df_train = _add_temperature_bins(df_train)

    markdown_cols = _get_markdown_columns(df_train)
    if markdown_cols:
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        df_train[markdown_cols] = imputer.fit_transform(df_train[markdown_cols])

    return df_train
