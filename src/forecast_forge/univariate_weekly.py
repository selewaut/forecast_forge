import argparse
import os

from forecast_forge.run_forecast import run_forecast
from pyspark.sql import SparkSession


parser = argparse.ArgumentParser(description="Run Walmart weekly forecast")
parser.add_argument(
    "--model",
    action="append",
    dest="models",
    help="Model(s) to run (can be repeated or comma-separated)",
)
args = parser.parse_args()


os.environ["NIXTLA_ID_AS_COL"] = "1"

experiment_path = os.getenv("FORECAST_EXPERIMENT_PATH", "testing/forecast")
run_name = os.getenv("FORECAST_RUN_NAME")
run_id = os.getenv("FORECAST_RUN_ID")

env_models = os.getenv("FORECAST_MODELS")
if args.models:
    raw = args.models
elif env_models:
    raw = env_models.split(",")
else:
    raw = [
        "StatsForecastBaselineWindowAverage",
        "StatsForecastBaselineSeasonalWindowAverage",
        "StatsForecastBaselineNaive",
        "StatsForecastBaselineSeasonalNaive",
        "StatsForecastAutoArima",
    ]

active_models = []
for m in raw:
    active_models.extend([x.strip() for x in m.split(",") if x.strip()])

if not active_models:
    parser.error("At least one model must be provided.")

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
    stride=3,
    metric="smape",
    train_predict_ratio=1,
    resample=False,
    active_models=active_models,
    experiment_path=experiment_path,
    run_name=run_name,
    run_id=run_id,
    use_case_name="walmart_daily",
)
