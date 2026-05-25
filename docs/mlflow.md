# MLflow Setup and Usage

Forecast Forge logs model evaluation metrics with MLflow. Local runs use the
tracking URI configured by MLflow in the current environment. In this checkout,
the expected local backend is:

```text
sqlite:///mlflow.db
```

## Start the UI

Start the MLflow UI from the repository root:

```sh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

If port `5000` is already in use:

```sh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## Experiments

The default weekly forecasting entrypoint logs to:

```text
testing/forecast
```

Use `FORECAST_EXPERIMENT_PATH` to override it for a run:

```sh
FORECAST_EXPERIMENT_PATH=walmart/weekly \
uv run spark-submit --master 'local[*]' src/forecast_forge/univariate_weekly.py
```

List local experiments:

```sh
uv run mlflow experiments search
```

## Run Names

Each evaluated model creates its own MLflow run. By default, runs are named with
the model name, for example:

```text
StatsForecastBaselineNaive
StatsForecastAutoArima
```

Use `FORECAST_RUN_NAME` to prefix all model runs from the same pipeline run:

```sh
FORECAST_RUN_NAME=baseline-v1 \
uv run spark-submit --master 'local[*]' src/forecast_forge/univariate_weekly.py
```

This produces run names like:

```text
baseline-v1-StatsForecastBaselineNaive
baseline-v1-StatsForecastAutoArima
```

Use `FORECAST_RUN_ID` when you need a stable pipeline-level identifier across
all model runs:

```sh
FORECAST_RUN_ID=baseline-v1-20260524 FORECAST_RUN_NAME=baseline-v1 \
uv run spark-submit --master 'local[*]' src/forecast_forge/univariate_weekly.py
```

The pipeline stores this value in MLflow tags as `run_id` and
`forecast_run_id`.

## Inspect Runs from the CLI

List runs in the default experiment:

```sh
uv run mlflow runs list --experiment-name testing/forecast
```

Inspect experiments and run counts with Python:

```sh
uv run python - <<'PY'
from mlflow.tracking import MlflowClient

client = MlflowClient()
for experiment in client.search_experiments():
    runs = client.search_runs([experiment.experiment_id])
    print(experiment.experiment_id, experiment.name, len(runs))
PY
```

## Outputs

MLflow stores aggregated metrics, model name tags, and run identifiers. The
pipeline also writes detailed evaluation records to the configured parquet
output path, which defaults to:

```text
weekly_evaluation_output
```
