from pathlib import Path

import pandas as pd
import yaml

from forecast_forge.loaders.base import DataLoadingConfig
from forecast_forge.loaders.schema import SchemaAdapter


class SchemaValidationError(ValueError):
    pass


def _find_datasets_dir() -> Path:
    candidates = [
        Path.cwd() / "datasets",
        Path(__file__).resolve().parents[3] / "datasets",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


class DatasetRegistry:
    def __init__(self, datasets_dir: str | Path | None = None):
        self.datasets_dir = Path(datasets_dir) if datasets_dir else _find_datasets_dir()

    def get(self, name: str) -> DataLoadingConfig:
        path = self.datasets_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found at {path}")
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in dataset '{name}': {e}") from e

        loader = data.pop("loader", None)
        if loader:
            data["loader_module"] = loader.get("module", "")
            data["loader_class"] = loader.get("class", "")

        try:
            return DataLoadingConfig(**data)
        except TypeError as e:
            raise TypeError(
                f"Invalid dataset config '{name}': {e}"
            ) from e

    @staticmethod
    def validate(df: pd.DataFrame, adapter: SchemaAdapter | None = None):
        required = ["unique_id", "ds", "y"]
        for col in required:
            if col not in df.columns:
                raise SchemaValidationError(
                    f"Required column '{col}' not found"
                )

        try:
            pd.to_datetime(df["ds"])
        except (ValueError, TypeError) as e:
            raise SchemaValidationError(
                f"Date column 'ds' is not parseable: {e}"
            ) from e

        if not pd.api.types.is_numeric_dtype(df["y"]):
            raise SchemaValidationError(
                f"Target column 'y' is not numeric (got {df['y'].dtype})"
            )

    @staticmethod
    def _build_adapter(schema: dict) -> SchemaAdapter:
        return SchemaAdapter(column_map=schema)
