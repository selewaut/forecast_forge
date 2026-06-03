import pathlib
from typing import Any, Union, List, Dict

import pandas as pd
import yaml
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer
from pyspark.sql import SparkSession, DataFrame

import importlib.resources as pkg_resources
from forecast_forge.forecaster import Forecaster


def run_forecast(
    spark: SparkSession,
    train_data: Union[str, pd.DataFrame, DataFrame],
    group_id: str,
    date_col: str,
    target: str,
    freq: str,
    prediction_length: int,
    backtest_periods: int,
    stride: int,
    metric: str = "smape",
    scoring_data: Union[str, pd.DataFrame, DataFrame] = None,
    scoring_output: str = None,
    evaluation_output: str = None,
    model_output: str = None,
    use_case_name: str = None,
    static_features: List[str] = None,
    dynamic_future: List[str] = None,
    dynamic_historical: List[str] = None,
    active_models: List[str] = None,
    accelerator: str = "cpu",
    backtest_retrain: bool = None,
    train_predict_ratio: int = None,
    resample: bool = False,
    experiment_path: str = None,
    run_name: str = None,
    run_id: str = None,
    conf: Union[str, Dict[str, Any], OmegaConf] = None,
) -> str:

    if isinstance(conf, dict):
        _conf = OmegaConf.create(conf)
    elif isinstance(conf, str):
        _yaml_conf = yaml.safe_load(pathlib.Path(conf).read_text())
        _conf = OmegaConf.create(_yaml_conf)
    elif isinstance(conf, BaseContainer):
        _conf = conf
    else:
        _conf = OmegaConf.create()

    base_conf = OmegaConf.create(
        pkg_resources.read_text("forecast_forge", "forecast_config.yaml")
    )
    _conf = OmegaConf.merge(base_conf, _conf)

    _data_conf = {}

    if isinstance(train_data, (pd.DataFrame, DataFrame)):
        _data_conf["train_data"] = train_data
    else:
        _conf["train_data"] = train_data

    _conf["group_id"] = group_id
    _conf["date_col"] = date_col
    _conf["target"] = target
    _conf["freq"] = freq
    _conf["prediction_length"] = prediction_length
    _conf["backtest_periods"] = backtest_periods
    _conf["stride"] = stride
    _conf["metric"] = metric
    _conf["resample"] = resample

    run_evaluation = True
    run_scoring = False

    if scoring_data is not None and scoring_output is not None:
        run_scoring = True
        _conf["scoring_output"] = scoring_output
        if isinstance(scoring_data, (pd.DataFrame, DataFrame)):
            _data_conf["scoring_data"] = scoring_data
        else:
            _conf["scoring_data"] = scoring_data
    if use_case_name is not None:
        _conf["use_case_name"] = use_case_name
    if active_models is not None:
        _conf["active_models"] = active_models
    if accelerator is not None:
        _conf["accelerator"] = accelerator
    if backtest_retrain is not None:
        _conf["backtest_retrain"] = backtest_retrain
    if train_predict_ratio is not None:
        _conf["train_predict_ratio"] = train_predict_ratio
    if experiment_path is not None:
        _conf["experiment_path"] = experiment_path
    if run_name is not None:
        _conf["run_name"] = run_name
    if evaluation_output is not None:
        _conf["evaluation_output"] = evaluation_output
    if model_output is not None:
        _conf["model_output"] = model_output
    if static_features is not None:
        _conf["static_features"] = static_features
    if dynamic_future is not None:
        _conf["dynamic_future"] = dynamic_future
    if dynamic_historical is not None:
        _conf["dynamic_historical"] = dynamic_historical
    if run_id is not None:
        _conf["run_id"] = run_id

    f = Forecaster(_conf, _data_conf, run_id=run_id, spark=spark)

    run_id = f.evaluate_score(evaluate=run_evaluation, score=run_scoring)
    return run_id
