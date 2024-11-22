import datetime
import functools
import pathlib
import uuid

import cloudpickle
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer

from forecast_forge.abstract_model import ForecastingRegressor
from forecast_forge.data_loader import load_data
from forecast_forge.data_processing import pre_process_data
from forecast_forge.model_registry import ModelRegistry


class Forecaster:
    def __init__(self, conf, data_conf, experiment_id=None, run_id=None):

        if isinstance(conf, BaseContainer):
            self.conf = conf
        elif isinstance(conf, dict):
            self.conf = OmegaConf.create(conf)
        elif isinstance(conf, str):
            _yaml_conf = yaml.safe_load(pathlib.Path(conf).read_text())
            self.conf = OmegaConf.create(_yaml_conf)
        else:
            raise ValueError("No configuration found!")

        if run_id:
            self.run_id = run_id
        else:
            self.run_id = str(uuid.uuid4())

        self.data_conf = data_conf
        self.model_registry = ModelRegistry(self.conf)

        if experiment_id:
            self.experiment_id = experiment_id
        elif self.conf.get("experiment_path"):
            self.experiment_id = self.set_mlflow_experiment()
        else:
            raise Exception("Set 'experiment_path' in configuration file")

        self.run_date = datetime.datetime.now()

    def set_mlflow_experiment(self):

        mlflow.set_experiment(self.conf["experiment_path"])
        experiment_id = (
            MlflowClient()
            .get_experiment_by_name(self.conf["experiment_path"])
            .experiment_id
        )
        return experiment_id

    def resolve_source(self, key: str):

        if self.data_conf:
            df_val = self.data_conf.get(key)
            if df_val is not None and isinstance(df_val, pd.DataFrame):
                return df_val
            else:
                df_val = self.load_data()
                return df_val

    def split_df_train_val(self, df: pd.DataFrame):
        train_df = df[
            df[self.conf["date_col"]]
            <= df[self.conf["date_col"]].max()
            - pd.DateOffset(weeks=self.conf["backtest_periods"])
        ]
        # Validate with data after the backtest months cutoff...
        val_df = df[
            df[self.conf["date_col"]]
            > df[self.conf["date_col"]].max()
            - pd.DateOffset(weeks=self.conf["backtest_periods"])
        ]
        return train_df, val_df

    @staticmethod
    def evaluate_one_local_model(
        pdf: pd.DataFrame, model: ForecastingRegressor
    ) -> pd.DataFrame:
        """
        A static method that evaluates a single local model using the provided pandas DataFrame and model.
        If the evaluation for a single group fails, it returns an empty DataFrame without failing the entire process.
        Parameters:
            pdf (pd.DataFrame): A pandas DataFrame.
            model (ForecastingRegressor): A ForecastingRegressor object.
        Returns: metrics_df (pd.DataFrame): A pandas DataFrame.
        """
        pdf = pdf.copy()
        pdf.reset_index(drop=True, inplace=True)
        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        split_date = pdf[model.params["date_col"]].max() - pd.DateOffset(
            weeks=model.params["backtest_periods"]
        )
        group_id = pdf[model.params["group_id"]].iloc[0]
        try:
            pdf = pdf.fillna(0)
            # Fix here
            pdf[model.params["target"]] = pdf[model.params["target"]].clip(0)
            metrics_df = model.backtest(
                pdf,
                start=split_date,
                group_id=group_id,
                stride=model.params["stride"],
            )
            return metrics_df
        except Exception:
            return pd.DataFrame(
                columns=[
                    model.params["group_id"],
                    "backtest_window_start_date",
                    "metric_name",
                    "metric_value",
                    "forecast",
                    "actual",
                    "model_pickle",
                ]
            )

    def score_models(self):
        "Scores models with provided config"
        for model_name in self.model_registry.get_active_model_keys():
            model_conf = self.model_registry.get_model_conf(model_name)
            print(f"Scoring model {model_name}")
            if model_conf["model_type"] == "local":
                self.score_local_model(model_conf)
            else:
                raise ValueError("Model type not supported (YET)")
            print("Finished scoring model {model_name}")

        print("Finished scoring all models")

    def load_data(self):
        df_train, df_test, df_features, df_stores = load_data()
        df_train, df_test = pre_process_data(
            df_train,
            df_test,
            df_features,
            df_stores,
            target_column=self.conf.get("target"),
            group_columns=[self.conf.get("group_id"), self.conf.get("date_col")],
            date_column=self.conf.get("date_col"),
        )
        # filter onyly 10 combinations time series to test the code
        # get the unique group_id from index (group_id, date)

        combinations = df_train.index.get_level_values(0).unique()[:10]

        df_train = df_train.loc[combinations]

        # count combinations in df_train
        print(
            f"Number of combinations in df_train: {df_train.index.get_level_values(0).nunique()}"
        )

        return df_train.reset_index()

    @staticmethod
    def score_one_local_model(
        self, pdf: pd.DataFrame, model: ForecastingRegressor
    ) -> pd.DataFrame:

        # convert to datetime.
        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        group_id = pdf[model.params["group_id"]].iloc[0]

        res_df, model_fitted = model.forecast(pdf, group_id=group_id)

        try:
            data = [
                group_id,
                res_df[model.params["date_col"]].to_numpy(),
                res_df["target"].to_numpy(),
                cloudpickle.dumps(model_fitted),
            ]
        except:
            data = [group_id, None, None, None]

        res_df = pd.DataFrame(
            columns=["group_id", "date", "target", "model_pickle"], data=[data]
        )
        return res_df

    def score_local_model(self, model_conf):
        # FIXME: df_test does not have actual validation data, its just the kaggle prediction data.
        src_df = self.resolve_source("train_data")
        model = self.model_registry.get_model(model_conf["name"])

        # train model for each combination.
        combinations_results = []
        for group_id in src_df["group_id"].unique():
            pdf = src_df[src_df["group_id"] == group_id]
            res_df = self.score_one_local_model(pdf, model)
            res_df["run_id"] = self.run_id
            res_df["model_name"] = model_conf["name"]
            res_df["model_conf"] = model_conf
            res_df["run_date"] = self.run_date
            mlflow.log_df(res_df, f"{model_conf['name']}_results")
            combinations_results.append(res_df)

        # combine all results
        all_results = pd.concat(combinations_results)
        mlflow.log_df(all_results, f"{model_conf['name']}_all_results")

    def evaluate_score(self, evaluate=True, score=False):
        if evaluate:
            self.evaluate_models()

        if score:
            self.score_models()
        print("Finished evaluate_score")
        return self.run_id

    def evaluate_models(self):
        """
        Trains and evaluates all models from the active models list.
        Parameters: self (Forecaster): A Forecaster object.
        """
        print("Starting evaluate_models")
        for model_name in self.model_registry.get_active_model_keys():
            print(f"Started evaluating {model_name}")
            try:
                model_conf = self.model_registry.get_model_conf(model_name)
                if model_conf["model_type"] == "local":
                    self.evaluate_local_model(model_conf)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
            print(f"Finished evaluating {model_name}")
        print("Finished evaluate_models")

    def evaluate_local_model(self, model_conf):
        """
        Evaluates a local model using the provided model configuration. It applies the Pandas UDF to the training data.
        It then logs the aggregated metrics and a few tags to MLflow.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        with mlflow.start_run(experiment_id=self.experiment_id):
            src_df = self.resolve_source("train_data")

            model = self.model_registry.get_model(model_conf["name"])

            combinations_results = []
            for group_id in src_df[model_conf["group_id"]].unique():
                # filter data for each group_id column
                pdf = src_df.loc[src_df[model_conf["group_id"]] == group_id]
                res_df = self.evaluate_one_local_model(pdf, model)
                res_df["run_id"] = self.run_id
                res_df["model_name"] = model_conf["name"]
                res_df["model_conf"] = model_conf
                res_df["run_date"] = self.run_date
                res_df["group_id"] = group_id
                # mlflow.log_artifact(res_df, f"{model_conf['name']}_results")

                combinations_results.append(res_df)

            all_results = pd.concat(combinations_results)

            #
            all_results["run_id"] = self.run_id
            all_results["run_date"] = self.run_date
            all_results["model"] = model_conf["name"]
            all_results["use_case"] = self.conf["use_case_name"]
            all_results["model_uri"] = ""

            # save all results
            # all_results.to_csv(
            #     f"results/{model_conf['name']}_all_results.csv", index=False
            # )

            # compute aggregated metrics
            agg_metrics = all_results.groupby("metric_name")["metric_value"].mean()

            # log aggregated metrics
            for metric_name, metric_value in agg_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
