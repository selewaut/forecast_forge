import logging
from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn import set_config  # To enable set_output

# Enable pandas output for all transformers globally
set_config(transform_output="pandas")

# Setup logging
logging.basicConfig(level=logging.INFO)


def merge_dataframes(
    df_main: pd.DataFrame,
    df_other: pd.DataFrame,
    on: List[str],
    merge_type: str = "left",
) -> pd.DataFrame:
    try:
        return pd.merge(df_main, df_other, on=on, how=merge_type)
    except KeyError as e:
        # logging.error(f"Error merging dataframes: {e}")
        raise


def negative_sales_to_zero(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # logging.info(f"Setting negative values in {target_col} to zero")
    df[target_col] = df[target_col].apply(lambda x: x if x > 0 else 0)
    return df


def add_numeric_temperature_bins(df, temp_col="temperature") -> pd.DataFrame:
    # logging.info(f"Creating temperature bins for column {temp_col}")
    bins_fahrenheit = [-np.inf, 40, 55, 70, 85, 95, np.inf]
    bin_labels_numeric = [0, 1, 2, 3, 4, 5]

    if temp_col not in df.columns:
        # logging.error(f"'{temp_col}' column not found in the DataFrame")
        raise ValueError(f"'{temp_col}' column not found")

    df["temp_bin"] = pd.cut(
        df[temp_col], bins=bins_fahrenheit, labels=bin_labels_numeric
    )
    df = df.drop(columns=[temp_col])
    return df


def generate_week_feature(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    logging.info(f"Generating week feature from {date_col}")
    if date_col not in df.columns:
        logging.error(f"'{date_col}' column not found in the DataFrame")
        raise ValueError(f"'{date_col}' column not found in the DataFrame")

    df[date_col] = pd.to_datetime(df[date_col])
    df["Week"] = df[date_col].dt.isocalendar().week
    return df


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
import pandas as pd


def build_preprocessing_pipeline(group_columns: List[str]) -> ColumnTransformer:
    logging.info("Building the preprocessing pipeline")

    markdown_features = [
        "markdown1",
        "markdown2",
        "markdown3",
        "markdown4",
        "markdown5",
    ]

    mean_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    markdown_imputer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    )

    # Include group_columns explicitly in passthrough
    preprocessor = ColumnTransformer(
        transformers=[
            # ("num", mean_imputer, numeric_features),
            ("markdown", markdown_imputer, markdown_features),
        ],
        remainder="passthrough",  # Ensure the group_columns are passed through unchanged
        verbose_feature_names_out=False,  # Use original column names in output
    )

    # Ensure the pipeline returns a DataFrame with column names
    preprocessor.set_output(transform="pandas")

    return preprocessor


def pre_process_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_features: pd.DataFrame,
    df_stores: pd.DataFrame,
    target_column: str,
    date_column: str,
    group_columns: List[str],
) -> (pd.DataFrame, pd.DataFrame):
    """
    Preprocess train and test data.

    df_train: pd.DataFrame - Training DataFrame.
    df_test: pd.DataFrame - Test DataFrame.
    df_features: pd.DataFrame - DataFrame with additional features.
    df_stores: pd.DataFrame - DataFrame with store details.
    target_column: str - Target column name.
    date_column: str - Date column name.
    group_columns: List[str] - Columns to be used for grouping (e.g., 'group_id', 'date').

    Returns:
    Tuple: Processed training and test DataFrames.
    """

    logging.info("Starting data preprocessing")

    # Merge features and stores
    df_train = merge_dataframes(
        df_train, df_features, on=["date", "store", "isholiday"]
    )
    df_train = merge_dataframes(df_train, df_stores, on=["store"])

    df_test = merge_dataframes(df_test, df_features, on=["date", "store", "isholiday"])
    df_test = merge_dataframes(df_test, df_stores, on=["store"])

    # Generate week feature
    df_train = generate_week_feature(df_train, date_col=date_column)
    df_test = generate_week_feature(df_test, date_col=date_column)

    # Handle negative sales
    df_train = negative_sales_to_zero(df_train, target_column)

    # Add temperature bins
    df_train = add_numeric_temperature_bins(df_train)

    # Preprocessing pipeline - group_columns passed to be retained
    pre_processing_pipeline = build_preprocessing_pipeline(group_columns)

    # Apply the transformation using the new set_output
    df_train_transformed = pre_processing_pipeline.fit_transform(df_train)

    # Log the columns of the transformed DataFrame to debug
    logging.info(f"Transformed columns: {df_train_transformed.columns}")

    # Check if group_columns exist after transformation
    missing_columns = [
        col for col in group_columns if col not in df_train_transformed.columns
    ]
    if missing_columns:
        raise KeyError(
            f"The following group columns are missing after transformation: {missing_columns}"
        )

    # Ensure group_columns are present in transformed DataFrame
    df_train_transformed.set_index(group_columns, inplace=True)

    logging.info("Data preprocessing completed")
    return df_train_transformed, df_test


if __name__ == "__main__":
    from forecast_forge.data_loader import load_data

    # Load data
    df_train, df_test, df_features, df_stores = load_data()

    # Preprocess data
    df_train_transformed, df_test = pre_process_data(
        df_train,
        df_test,
        df_features,
        df_stores,
        target_column="weekly_sales",
        date_column="date",
        group_columns=["group_id", "date"],
    )

    print("finished")
