import mlflow
from mlflow.tracking import MlflowClient
import scipy as sp
from forecast_forge.run_forecast import run_forecast

active_models = [
    "StatsForecastBaselineWindowAverage",
    # "StatsForecastBaselineSeasonalWindowAverage",
    # "StatsForecastBaselineNaive",
    # "StatsForecastBaselineSeasonalNaive",
    #    "StatsForecastAutoArima",
    # "StatsForecastAutoETS",
    # "StatsForecastAutoCES",
    # "StatsForecastAutoTheta",
    # "StatsForecastTSB",
    # "StatsForecastADIDA",
    # "StatsForecastIMAPA",
    # "StatsForecastCrostonClassic",
    # "StatsForecastCrostonOptimized",
    # "StatsForecastCrostonSBA",
]


mlflow.set_experiment("experiments/testing/forecast")
experiment_id = (
    MlflowClient().get_experiment_by_name("experiments/testing/forecast").experiment_id
)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("forecast").getOrCreate()

run_forecast(
    spark=spark,
    train_data="train_data",
    evaluation_output="weekly_evaluation_output",
    group_id="group_id",
    date_col="date",
    target="weekly_sales",
    freq="W",
    prediction_length=12,
    backtest_periods=12,
    stride=12,
    metric="smape",
    train_predict_ratio=1,
    resample=False,
    active_models=active_models,
    experiment_path=f"testing/forecast",
    use_case_name="walmart_daily",
)
