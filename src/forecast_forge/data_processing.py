import pandas as pd
import pandas as pd
import numpy as np
from typing import List


def merge_features(
    df_train: pd.DataFrame, df_features: pd.DataFrame, on: List[str]
) -> pd.DataFrame:
    """
    df_train: pd.DataFrame - DataFrame with the target variable, can contain multiple time series.
    df_features: pd.DataFrame - DataFrame with the features to merge with df_train.
    """
    return pd.merge(df_train, df_features, on=on, how="left")


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


# build pipeline using SKlearn Pipeline and ColumnTransformer for preprocessing using prior functions
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessing_pipeline():
    """
    Build a preprocessing pipeline using ColumnTransformer from scikit-learn.

    Returns:
    ColumnTransformer: Preprocessing pipeline for feature transformation.
    """
    # Define numerical and categorical features
    numeric_features = ["temperature", "fuel_price", "cpi", "unenployment"]

    # Create a preprocessing pipeline for numerical features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    # Combine the numerical and categorical preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    return preprocessor
