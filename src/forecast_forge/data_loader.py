from json import load
import pandas as pd
from forecast_forge.config import TRAIN_PATH, TEST_PATH, FEATURES_PATH, STORES_PATH


def load_data():

    # if paths exist, load data

    if not TRAIN_PATH.exists():
        # download data
        from forecast_forge.data import download_data

        print("Downloading data")
        download_data()

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df_features = pd.read_csv(FEATURES_PATH)
    df_stores = pd.read_csv(STORES_PATH)

    # lower case column names
    df_train.columns = df_train.columns.str.lower()
    df_test.columns = df_test.columns.str.lower()
    df_features.columns = df_features.columns.str.lower()
    df_stores.columns = df_stores.columns.str.lower()

    df_train["date"] = pd.to_datetime(df_train["date"])
    df_test["date"] = pd.to_datetime(df_test["date"])
    df_features["date"] = pd.to_datetime(df_features["date"])

    # combine store dept columns
    df_train["store_dept"] = (
        df_train["store"].astype(str) + "_" + df_train["dept"].astype(str)
    )
    df_test["store_dept"] = (
        df_test["store"].astype(str) + "_" + df_test["dept"].astype(str)
    )

    return df_train, df_test, df_features, df_stores


if __name__ == "__main__":
    load_data()
