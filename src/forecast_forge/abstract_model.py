from abc import abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Union
from sklearn import metrics
from sklearn.base import BaseEstimator, RegressorMixin
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)


class ForecastingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params):
        self.params = params
        self.freq = params["freq"]

        if self.freq == "W":
            self.prediction_length_offeset = pd.offsets.DateOffset(
                weeks=params["prediction_length"]
            )
            self.one_ts_offset = pd.offsets.DateOffset(weeks=1)
        elif self.freq == "D":
            self.prediction_length_offeset = pd.offsets.DateOffset(
                days=params["prediction_length"]
            )
            self.one_ts_offset = pd.offsets.DateOffset(days=1)
        elif self.freq == "M":
            self.prediction_length_offeset = pd.offsets.DateOffset(
                months=params["prediction_length"]
            )
            self.one_ts_offset = pd.offsets.DateOffset(months=1)
        else:
            raise ValueError("Invalid frequency")

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def backtest(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp,
        group_id: Union[str, int],
        stride: int = None,
    ):
        """
        Backtest a model on a time series DataFrame.

        Args:
        df (pd.DataFrame): DataFrame with time series data.
        start (pd.Timestamp): Start date for the backtest.
        group_id (Union[str, int]): Column name or index of the column containing the group ID.
        stride (int): Number of steps to move the window forward. Default is None.

        Returns:
        pd.DataFrame: DataFrame with the backtest results.
        """
        if stride is None:
            stride = self.params["prediction_length"]

        # stride_offset =
        if self.freq == "W":
            stride_offset = pd.offsets.DateOffset(weeks=stride)
        elif self.freq == "D":
            stride_offset = pd.offsets.DateOffset(days=stride)
        elif self.freq == "M":
            stride_offset = pd.offsets.DateOffset(months=stride)
        else:
            raise ValueError("Invalid frequency")

        df = df.copy().sort_values(by=[self.params["date_col"]])
        end_date = df[self.params["date_col"]].max()
        curr_date = start + self.one_ts_offset

        results = []

        while (
            curr_date + self.prediction_length_offeset <= end_date + self.one_ts_offset
        ):
            _df = df[df[self.params["date_col"]] <= np.datetime64(curr_date)]
            actuals_df = df[
                (df[self.params["date_col"]] >= np.datetime64(curr_date))
                & (
                    df[self.params["date_col"]]
                    < np.datetime64(curr_date) + self.prediction_length_offeset
                )
            ]

            metrics = self.calculate_metrics(_df, actuals_df, curr_date)

            if isinstance(metrics, dict):
                evaluation_results = [
                    (
                        group_id,
                        metrics["curr_date"],
                        metrics["metric_name"],
                        metrics["metric_value"],
                        metrics["forecast"],
                        metrics["actual"],
                        metrics["model_pickle"],
                    )
                ]

                results.extend(evaluation_results)

            elif isinstance(metrics, list):
                results.extend(metrics)

            curr_date += stride_offset

        res_df = pd.DataFrame(
            results,
            columns=[
                self.params["group_id"],
                "backtest_window_start_date",
                "metric_name",
                "metric_value",
                "forecast",
                "actual",
                "model_pickle",
            ],
        )

        return res_df

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date
    ) -> Dict[str, Union[str, float, bytes]]:
        """
        Calculates the metrics using the provided historical DataFrame, validation DataFrame, current date.
        Parameters:
            self (Forecaster): A Forecaster object.
            hist_df (pd.DataFrame): A pandas DataFrame.
            val_df (pd.DataFrame): A pandas DataFrame.
            curr_date: A pandas Timestamp object.
        Returns: metrics (Dict[str, Union[str, float, bytes]]): A dictionary specifying the metrics.
        """
        pred_df, model_fitted = self.predict(hist_df, val_df)

        actual = val_df[self.params["target"]].to_numpy()
        forecast = pred_df[self.params["target"]].to_numpy()

        if self.params["metric"] == "smape":
            smape = MeanAbsolutePercentageError(symmetric=True)
            metric_value = smape(actual, forecast)
        elif self.params["metric"] == "mape":
            mape = MeanAbsolutePercentageError(symmetric=False)
            metric_value = mape(actual, forecast)
        elif self.params["metric"] == "mae":
            mae = MeanAbsoluteError()
            metric_value = mae(actual, forecast)
        elif self.params["metric"] == "mse":
            mse = MeanSquaredError(square_root=False)
            metric_value = mse(actual, forecast)
        elif self.params["metric"] == "rmse":
            rmse = MeanSquaredError(square_root=True)
            metric_value = rmse(actual, forecast)
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")

        return {
            "curr_date": curr_date,
            "metric_name": self.params["metric"],
            "metric_value": metric_value,
            "forecast": pred_df[self.params["target"]].to_numpy("float"),
            "actual": val_df[self.params["target"]].to_numpy(),
            "model_pickle": model_fitted,
        }
