# arima models


from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima, model_selection
from sklearn.metrics import mean_squared_error
from ray.util.multiprocessing import Pool


from forecast_forge.data_processing import (
    add_numeric_temperature_bins,
    merge_features,
    negative_sales_to_zero,
    merge_stores,
    build_preprocessing_pipeline,
)
from forecast_forge.data_loader import load_data


def pre_process_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_features: pd.DataFrame,
    target_column: str,
    date_column: str,
):
    df_train = df_train.copy()

    df_train[date_column] = pd.to_datetime(df_train[date_column])
    df_test[date_column] = pd.to_datetime(df_test[date_column])
    df_features[date_column] = pd.to_datetime(df_features[date_column])

    # lower case column names
    df_train.columns = df_train.columns.str.lower()
    df_test.columns = df_test.columns.str.lower()
    df_features.columns = df_features.columns.str.lower()
    # merge features
    df_train = merge_features(
        df_train, df_features, on=["date", "store", "dept", "isholiday"]
    )
    df_train = merge_stores(df_train, df_stores, on=["store"])

    # general imputting.
    df_train = negative_sales_to_zero(df_train, target_column)
    df_train = add_numeric_temperature_bins(df_train)

    # pre-processing pipeline
    pre_processing_pipeline = build_preprocessing_pipeline()
    df_train = pre_processing_pipeline.fit_transform(df_train)

    # create ARIMA model tuning parameters for each series.

    return df_train


def fit_arima(ts, store_name):
    # Automatically find the best ARIMA model for each store's time series
    model = auto_arima(
        ts,
        start_p=1,
        start_q=1,
        start_P=1,
        start_Q=1,
        max_p=5,
        max_q=5,
        max_P=5,
        max_Q=5,
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        D=10,
        max_D=10,
        error_action="ignore",
    )
    return store_name, model


def forecast(model, periods=5):
    # Forecast for 'periods' steps into the future
    forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    return forecast, conf_int


def fit_models_for_stores(df):
    """
    Parallelized ARIMA fitting using Ray's Pool for parallelism
    df: Polars DataFrame with each store's time series as columns
    """
    with Pool() as pool:
        results = pool.starmap(
            fit_arima, [(df[col].to_numpy(), col) for col in df.columns]
        )
    models = {store: model for store, model in results}
    return models


def split_train_val(df_train, val_size=39):
    """
    Split df_train into df_train_split and df_val where df_val contains the last val_size dates
    df_train: Polars DataFrame with time series data
    val_size: Number of rows (dates) to use for validation set (default is 39)
    """
    # Ensure the DataFrame is sorted by date
    df_train = df_train.sort("date")

    # get max_date by combination

    df_max_date = df_train.groupby(["store", "dept"])["date"].max()

    # substract val_size from max_date

    df_max_date = df_max_date - pd.Timedelta(days=val_size)

    # reset the index and rename date

    df_max_date = df_max_date.reset_index().rename(columns={"date": "max_date"})

    # merge the max_date with the original df_train
    df_train = df_train.merge(df_max_date, on=["store", "dept"], how="left")

    # filter the data based on the max_date

    df_val = df_train[df_train["date"] > df_train["max_date"]]
    df_train = df_train[df_train["date"] <= df_train["max_date"]]

    return df_train, df_val


def train(df_train, df_test, df_features, df_stores):

    df_train = pre_process_data(
        df_train,
        df_test,
        df_features,
        target_column="weekly_sales",
        date_column="date",
    )

    df_wide = df_train.pivot(values="sales", index="date", columns=["store", "dept"])

    # split the data into train and test.Forecast horizon is 39 weeks for each combination.
    df_train_split, df_val = split_train_val(df_wide)

    # fit the ARIMA models for each store and department combination
    models = fit_models_for_stores(df_train_split)

    # evaluate the models on the validation set
    val_predictions = {}
    for store_dept, model in models.items():
        store, dept = store_dept
        val_predictions[store_dept] = forecast(model)

    # calculate error metrics
    val_errors = {}
    for store_dept, (forecast, conf_int) in val_predictions.items():
        store, dept = store_dept
        val_errors[store_dept] = mean_squared_error(
            df_val[(store, dept)].to_numpy(), forecast
        )

    # export the models and errors
    return models, val_errors


if __name__ == "__main__":
    df_train, df_test, df_features, df_stores = load_data()

    models, val_errors = train()
    # save errors

    # get average error by store total volumne
    mean_error = {
        store: np.mean(
            [val_errors[(store, dept)] for dept in df_train["dept"].unique()]
        )
        for store in df_train["store"].unique()
    }

    # get the store with the lowest error

    # export model errors in csv file

    print(mean_error)
