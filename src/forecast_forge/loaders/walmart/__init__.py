import os
import zipfile
from pathlib import Path

import kaggle
import pandas as pd

from forecast_forge.loaders.base import BaseDataLoader, DataLoadingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "walmart_sales_forecasting"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
STORES_PATH = DATA_DIR / "stores.csv"

WALMART_SCHEMA = {
    "unique_id": "group_id",
    "ds": "date",
    "y": "weekly_sales",
    "exogenous": [
        "isholiday",
        "temperature",
        "fuel_price",
        "cpi",
        "type",
        "size",
    ],
}

DEFAULT_CONFIG = DataLoadingConfig(
    loader_module="forecast_forge.loaders.walmart",
    loader_class="WalmartDataLoader",
    files={
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "features_path": str(FEATURES_PATH),
        "stores_path": str(STORES_PATH),
    },
    schema=WALMART_SCHEMA,
    download={
        "source": "kaggle",
        "competition": "walmart-recruiting-store-sales-forecasting",
    },
)


class WalmartDataLoader(BaseDataLoader):
    def __init__(self, config: DataLoadingConfig | None = None):
        super().__init__(config or DEFAULT_CONFIG)

    def download(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            competition="walmart-recruiting-store-sales-forecasting",
            path=DATA_DIR,
            force=True,
        )
        zip_path = DATA_DIR / "walmart-recruiting-store-sales-forecasting.zip"
        if zip_path.exists():
            self._unzip_files(zip_path, DATA_DIR)
            zip_path.unlink()

    @staticmethod
    def _unzip_files(zip_path, extract_to):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            for file in zip_ref.namelist():
                if file.endswith(".zip"):
                    nested_zip_path = os.path.join(extract_to, file)
                    WalmartDataLoader._unzip_files(nested_zip_path, extract_to)
                    os.remove(nested_zip_path)

    def load(self) -> pd.DataFrame:
        if not TRAIN_PATH.exists():
            self.download()

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

        return df_train
