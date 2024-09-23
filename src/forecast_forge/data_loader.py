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
    return df_train, df_test, df_features, df_stores


if __name__ == "__main__":
    load_data()
