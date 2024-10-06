from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def merge_features(
    df_train: pd.DataFrame, df_features: pd.DataFrame, on: List[str]
) -> pd.DataFrame:
    """
    df_train: pd.DataFrame - DataFrame with the target variable, can contain multiple time series.
    df_features: pd.DataFrame - DataFrame with the features to merge with df_train.
    """
    return pd.merge(df_train, df_features, on=on, how="left")


def merge_stores(
    df_train: pd.DataFrame, df_stores: pd.DataFrame, on: List[str]
) -> pd.DataFrame:
    """
    df_train: pd.DataFrame - DataFrame with the target variable, can contain multiple time series.
    df_stores: pd.DataFrame - DataFrame with the store information to merge with df_train.
    """
    return pd.merge(df_train, df_stores, on=on, how="left")


def negative_sales_to_zero(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    df: pd.DataFrame - DataFrame with the target variable.
    target_col: str - Name of the target column.
    """
    df[target_col] = df[target_col].apply(lambda x: x if x > 0 else 0)
    return df


def null_imputation_zero(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    cols: List[str] - List of columns to impute.

    """
    for col in cols:
        df[col] = df[col].fillna(0)
    return df


def add_numeric_temperature_bins(df, temp_col="temperature"):
    """
    Adds a 'Temp_Bin' column with numeric labels to the DataFrame based on temperature ranges in Fahrenheit.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a column containing temperature values in Fahrenheit.
    temp_col (str): The column name in the DataFrame that contains temperature values. Default is 'Avg_Temperature_F'.

    Returns:
    pd.DataFrame: DataFrame with an additional column 'Temp_Bin' categorizing temperature ranges numerically.
    """
    # Define temperature ranges in Fahrenheit
    bins_fahrenheit = [-np.inf, 40, 55, 70, 85, 95, np.inf]  # Temperature ranges

    # Numeric labels for each bin (e.g., 0 = Very Cold, 1 = Cold, etc.)
    bin_labels_numeric = [0, 1, 2, 3, 4, 5]  # Numeric labels for each bin

    # Ensure the temperature column exists in the DataFrame
    if temp_col not in df.columns:
        raise ValueError(f"'{temp_col}' column not found in the DataFrame")

    # Apply the temperature binning with numeric labels
    df["temp_bin"] = pd.cut(
        df[temp_col], bins=bins_fahrenheit, labels=bin_labels_numeric
    )
    # drop the original temperature column
    df = df.drop(columns=[temp_col])
    return df


def generate_week_feature(df, date_col="date"):
    """
    Generates a 'Week' feature from the date column in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a date column.
    date_col (str): The name of the date column in the DataFrame. Default is 'Date'.

    Returns:
    pd.DataFrame: DataFrame with an additional 'Week' feature extracted from the date column.
    """
    # Ensure the date column exists in the DataFrame
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column not found in the DataFrame")

    # Convert the date column to datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract the week number from the date column
    df["Week"] = df[date_col].dt.isocalendar().week

    return df


def build_preprocessing_pipeline():
    """
    Build a preprocessing pipeline using ColumnTransformer from scikit-learn.

    Returns:
    ColumnTransformer: Preprocessing pipeline for feature transformation.
    """
    # Define numerical and categorical features
    numeric_features = ["cpi", "unemployment"]

    markdown_features = [
        "markdown1",
        "markdown2",
        "markdown3",
        "markdown4",
        "markdown5",
    ]

    mean_imputer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    markdown_imputer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    # apply both imputers to the respective columns

    preprocessor = ColumnTransformer(
        transformers=[
            # ("num", mean_imputer, numeric_features),
            ("markdown", markdown_imputer, markdown_features),
        ]
    )

    return preprocessor


def pre_process_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_features: pd.DataFrame,
    df_stores: pd.DataFrame,
    target_column: str,
    date_column: str,
):

    # merge features
    df_train = merge_features(df_train, df_features, on=["date", "store", "isholiday"])
    df_train = merge_stores(df_train, df_stores, on=["store"])

    df_test = merge_features(df_test, df_features, on=["date", "store", "isholiday"])
    df_test = merge_stores(df_test, df_stores, on=["store"])

    # generate week feature
    df_train = generate_week_feature(df_train, date_col=date_column)
    df_test = generate_week_feature(df_test, date_col=date_column)

    # general imputting.
    df_train = negative_sales_to_zero(df_train, target_column)
    df_train = add_numeric_temperature_bins(df_train)

    # pre-processing pipeline
    pre_processing_pipeline = build_preprocessing_pipeline()
    df_train = pre_processing_pipeline.fit_transform(df_train)

    df_train.set_index(["group_id", "date"], inplace=True)

    return df_train, df_test


if __name__ == "__main__":
    from forecast_forge.data_loader import load_data

    df_train, df_test, df_features, df_stores = load_data()

    df_train = pre_process_data(
        df_train,
        df_test,
        df_features,
        df_stores,
        target_column="weekly_sales",
        date_column="date",
    )
